import bindsnet, pickle, torch, numpy as np, cv2, os, glob

device = 'cuda'

#files = glob.glob('debug/*.jpg', recursive=True)
#for f in files:
#    os.remove(f)

display_step = 15

nb_hidden = 1000

out_width = 30
nb_output = out_width * out_width

LR = .00003
pat_LR = 0.002
norm_speed = 0.005

sim_duration = 100

inh_level = -1.0

c1_wmax = 2.0
c3_wmax = 2.0

print('Importing MNIST...')
train_set_x = pickle.load(open('mnist_x.pkl', 'rb'))
train_set_y = pickle.load(open('mnist_y.pkl', 'rb')).astype(np.int64)

print('Building network...')
model = bindsnet.network.Network(dt=1, batch_size=1, learning=True)

m_input = bindsnet.network.nodes.Input(784, shape=(1, 784), traces=True)
m_hidden = bindsnet.network.nodes.LIFNodes(nb_hidden, traces=True, tc_decay=500)
m_output = bindsnet.network.nodes.LIFNodes(nb_output, traces=True, tc_decay=500)

model.add_layer(m_input, 'Input')
model.add_layer(m_hidden, 'Hidden')
model.add_layer(m_output, 'Output')
# titi
w_inh = np.zeros((nb_output, nb_output), dtype=np.float32)
for i in range(nb_output):
    x_i = i%out_width
    y_i = i//out_width
    for j in range(nb_output):
        x_j = j%out_width
        y_j = j//out_width
        dx = x_i - x_j
        dy = y_i - y_j
        d = np.sqrt(dx*dx + dy*dy)
        w_inh[i, j] = 1 - 0.15*d
w_inh[w_inh<0] = 0
w_inh *= inh_level
np.fill_diagonal(w_inh, 0.0)

print('Avg inh=', np.mean(w_inh))

my_w = np.random.normal(loc=0, scale=c1_wmax*0.5, size=(784, nb_hidden))
con1 = bindsnet.network.topology.Connection(source=m_input, target=m_hidden, nu=[c1_wmax*LR, c1_wmax*LR], update_rule=bindsnet.learning.PostPre, w=torch.Tensor(my_w), wmin=-c1_wmax, wmax=c1_wmax, weight_decay=0.00001)

my_w = np.random.normal(loc=0, scale=c3_wmax*0.5, size=(nb_hidden, nb_output))
con3 = bindsnet.network.topology.Connection(source=m_hidden, target=m_output, nu=[c3_wmax*LR, c3_wmax*LR], update_rule=bindsnet.learning.PostPre, w=torch.Tensor(my_w), wmin=-c3_wmax, wmax=c3_wmax, weight_decay=0.00001)

con4 = bindsnet.network.topology.Connection(source=m_output, target=m_output, nu=[0.0, 0.0], update_rule=bindsnet.learning.NoOp, w=torch.Tensor(w_inh), wmin=-1000.0, wmax=1000.0)

model.add_connection(con1, source='Input', target='Hidden')
#model.add_connection(con2, source='Hidden', target='Hidden')
model.add_connection(con3, source='Hidden', target='Output')
model.add_connection(con4, source='Output', target='Output')

print(bindsnet.analysis.visualization.summary(model))

out_spikes = bindsnet.network.monitors.Monitor(model.layers['Output'], state_vars=["s"], time=sim_duration)
model.add_monitor(out_spikes, name="out_spikes")

model.to(device)

poisson = bindsnet.encoding.PoissonEncoder(sim_duration, 1, )

train_size = len(train_set_x)

acc = 0.0

best_acc = 0.0

def first_spikes(s):
    nbn = s.shape[1]
    nbt = s.shape[0]

    first = np.full(nbn, np.exp(-nbt*0.1), dtype=np.float32)

    for i in range(nbn):
        for j in range(nbt):
            if s[j, i]:
                first[i] = np.exp(-j*0.05)
                #first[i] = -j
                break
    return first

spikes_decay = np.zeros(shape=(sim_duration), dtype=np.float32)

for i in range(sim_duration):
    spikes_decay[i] = np.exp(-i*0.02)

patterns = np.zeros(shape=(10, nb_output), dtype=np.float32)
noise = np.zeros((nb_output), dtype=np.float32)

for e in range(train_size):
    label = train_set_y[e]
    spikes = poisson(torch.as_tensor(train_set_x[e]*100.0)).to(device)

    model.reset_state_variables()
    model.run(inputs={'Input' : spikes}, time=sim_duration, input_time_dim=1)

    spike_record = out_spikes.get("s").squeeze().to('cpu').numpy().astype(np.float32)

    spike_sum = first_spikes(spike_record)

    #for i in range(nb_output):
    #    spike_record[:, i] *= spikes_decay

    #sum spikes temporally
    #spike_sum = np.sum(spike_record, axis=0, dtype=np.float32)

    if e%display_step==0:
        print('avg time=', np.mean(spike_sum), ' fast =', np.max(spike_sum), ' slow', np.min(spike_sum))

    #remove background static noise
    #noise = noise*0.99 + 0.01 * spike_sum
    #spike_sum -= noise

    #normalize total numbers of spikes
    #spike_sum -= np.min(spike_sum)
    #spike_sum *= 1.0 / (.0001 + np.mean(spike_sum))
    #spike_sum *= 0.1

    if e%display_step==0:
        img = 0.0 + 10.0 * 128.0 * spike_sum
        img = np.reshape(img, (out_width, out_width))
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('debug/rates'+str(int(train_set_y[e]))+'.jpg', img)

        for i in range(10):
            img = 0.0 + 20.0 * 128.0 * patterns[i]
            img = np.reshape(img, (out_width, out_width))
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('debug/pattern'+str(i)+'.jpg', img)

    #first_spike = first_spikes(spike_record)
    #first_spike -= np.mean(first_spike)

    best = 10000000.0
    result = -1
    for i in range(10):
        d = np.sum(np.square(patterns[i, :] - spike_sum))
        #d = np.sum(np.abs(patterns[i, :] - spike_sum))
        #if e%display_step==0:
        #    print(i, d)
        if d < best:
            best = d
            result = i
    if result == label:
        acc = acc*0.99 + 0.01
    else:
        acc = acc*0.99

    if e%display_step==0:
        if acc>best_acc:
            #model.save('best_MNIST.snn')
            best_acc = acc
        print(100.0*e/train_size,'%', e, '/', train_size)
        print('result=', result, '   ground=', train_set_y[e])
        print('accuracy=', 100.0*acc, '%', 'best=', 100.0*best_acc)

        print('w1=', torch.mean(torch.abs(con1.w)).to('cpu').numpy(), 'w3=', torch.mean(torch.abs(con3.w)).to('cpu').numpy())

    patterns[label, :] = (1.0 - pat_LR) * patterns[label, :] + pat_LR * spike_sum

    # slow re-normalizing
    if True:
        m1 = torch.mean(torch.abs(con1.w))
        m3 = torch.mean(torch.abs(con3.w))

        ratio1 = (1.0 - norm_speed) + norm_speed * 0.5 * con1.wmax / m1
        ratio3 = (1.0 - norm_speed) + norm_speed * 0.5 * con3.wmax / m3

        con1.w *= ratio1
        con3.w *= ratio3
