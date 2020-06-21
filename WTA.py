import bindsnet, pickle, torch, numpy as np, cv2, os, glob
from tqdm import tqdm
from torchvision import transforms

device = 'cuda'
gpu = True

files = glob.glob('debug/*.jpg', recursive=True)
for f in files:
    os.remove(f)

poisson_intensity = 100.0
batch_size = 16

display_step = 2

nb_hidden = 1000

out_width = 30
nb_output = out_width * out_width

LR = 1e-6   # 1e-6
pat_LR = 1e-4
norm_speed = np.power(0.005, batch_size)

sim_duration = 100

inh_level = -1.0

c1_wmax = 2.0
c3_wmax = 2.0

print('Building network...')
model = bindsnet.network.Network(dt=1, batch_size=batch_size, learning=True)

m_input = bindsnet.network.nodes.Input(784, shape=(1, 784), traces=True)
m_hidden = bindsnet.network.nodes.LIFNodes(nb_hidden, traces=True, tc_decay=500)
m_output = bindsnet.network.nodes.LIFNodes(nb_output, traces=True, tc_decay=500)

model.add_layer(m_input, 'Input')
model.add_layer(m_hidden, 'Hidden')
model.add_layer(m_output, 'Output')

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
con1 = bindsnet.network.topology.Connection(source=m_input, target=m_hidden, nu=[c1_wmax*LR, c1_wmax*LR], update_rule=bindsnet.learning.PostPre, w=torch.Tensor(my_w), wmin=-c1_wmax, wmax=c1_wmax, weight_decay=0.00001, reduction=torch.sum)

my_w = np.random.normal(loc=0, scale=c3_wmax*0.5, size=(nb_hidden, nb_output))
con3 = bindsnet.network.topology.Connection(source=m_hidden, target=m_output, nu=[c3_wmax*LR, c3_wmax*LR], update_rule=bindsnet.learning.PostPre, w=torch.Tensor(my_w), wmin=-c3_wmax, wmax=c3_wmax, weight_decay=0.00001, reduction=torch.sum)

con4 = bindsnet.network.topology.Connection(source=m_output, target=m_output, nu=[0.0, 0.0], update_rule=bindsnet.learning.NoOp, w=torch.Tensor(w_inh), wmin=-1000.0, wmax=1000.0, reduction=torch.sum)

model.add_connection(con1, source='Input', target='Hidden')
model.add_connection(con3, source='Hidden', target='Output')
model.add_connection(con4, source='Output', target='Output')

print(bindsnet.analysis.visualization.summary(model))

out_spikes = bindsnet.network.monitors.Monitor(model.layers['Output'], state_vars=["s"], time=sim_duration)
model.add_monitor(out_spikes, name="out_spikes")

model.to(device)

# Load MNIST data.
train_dataset = bindsnet.datasets.MNIST(
    bindsnet.encoding.PoissonEncoder(time=sim_duration, dt=1),
    None,
    root=os.path.join("data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * poisson_intensity)]
    ),
)

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=gpu
)

acc = 0.0
best_acc = 0.0

def first_spikes(s):
    nbt = s.shape[0]
    nbn = s.shape[1]

    first = np.full(nbn, np.exp(-nbt*0.1), dtype=np.float32)

    for i in range(nbn):
        for j in range(nbt):
            if s[j, i]:
                first[i] = np.exp(-j*0.05)
                break
    return first


patterns = np.zeros(shape=(10, nb_output), dtype=np.float32)
noise = np.zeros((nb_output), dtype=np.float32)

for step, batch in enumerate(tqdm(dataloader)):
    i = batch["encoded_image"].view(batch_size, sim_duration, 1, 784)
    i = i.permute(1, 0, 2, 3)
    inputs = {"Input": i}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    label = batch["label"]

    model.reset_state_variables()
    model.run(inputs=inputs, time=sim_duration, input_time_dim=1)

    spike_record = out_spikes.get("s").squeeze().to('cpu').numpy().astype(np.float32)

    for e in range(batch_size):
        spike_sum = first_spikes(spike_record[:, e, :])
        if e==0:
            print('avg time=', np.mean(spike_sum), ' fast =', np.max(spike_sum), ' slow', np.min(spike_sum))

        best = 10000000.0
        result = -1
        for i in range(10):
            d = np.sum(np.square(patterns[i, :] - spike_sum))
            if d < best:
                best = d
                result = i
        if result == label[e]:
            acc = acc*0.995 + 0.005
        else:
            acc = acc*0.995
        patterns[label[e], :] = (1.0 - pat_LR) * patterns[label[e], :] + pat_LR * spike_sum

    if step % display_step==0:
        if acc>best_acc:
            best_acc = acc
        print('\n accuracy=', 100.0*acc, '%', 'best=', 100.0*best_acc)
        print('w1=', torch.mean(torch.abs(con1.w)).to('cpu').numpy(), 'w3=', torch.mean(torch.abs(con3.w)).to('cpu').numpy())

        img = 0.0 + 10.0 * 128.0 * spike_sum
        img = np.reshape(img, (out_width, out_width))
        img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('debug/rates'+str(int(label[e]))+'.jpg', img)

        for i in range(10):
           img = 0.0 + 20.0 * 128.0 * patterns[i]
           img = np.reshape(img, (out_width, out_width))
           img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
           cv2.imwrite('debug/pattern'+str(i)+'.jpg', img)


    # slow re-normalizing
    if True:
        m1 = torch.mean(torch.abs(con1.w))
        m3 = torch.mean(torch.abs(con3.w))

        ratio1 = (1.0 - norm_speed) + norm_speed * 0.5 * con1.wmax / m1
        ratio3 = (1.0 - norm_speed) + norm_speed * 0.5 * con3.wmax / m3

        con1.w *= ratio1
        con3.w *= ratio3
