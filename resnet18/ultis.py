import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class_cindex = {'0': 'bdsr', '1': 'csl', '2': 'fwq', '3': 'gj', '4': 'htj', '5': 'hy', '6': 'lgq', '7': 'lqs', '8': 'lx', '9': 'mf', '10': 'mzd', 
                '11': 'oyx', '12': 'sgt', '13': 'shz', '14': 'smh', '15': 'wxz', '16': 'wzm', '17': 'yyr', '18': 'yzq', '19': 'zmf'}

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        writer = SummaryWriter("logs")
        writer.add_image("dataloader", make_grid(images, nrow=16), 1)
        writer.close()
        break

def train(model, data_loader, optimizer, loss_fn):
    for data in data_loader:
        imgs, label = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            label = label.cuda()
        output = model(imgs)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, data_loader, loss_fn, total_val_loss, accuracy):
    for data in data_loader:
            imgs, label = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            output = model(imgs)
            loss = loss_fn(output, label)
            total_val_loss += loss.item()
            accuracy = (output.argmax(1) == label).sum()
            total_accuracy += accuracy

def plot_preds(net, images_dir, transfrom, class_index=class_cindex, device="cpu"):
    images = []
    labels = []
    for i in range(20):
        img = Image.open(images_dir + str(i) + ".jpg").convert("RGB")
        label = i
        img = transfrom(img)
        images.append(img)
        labels.append(label)

    images = torch.stack(images, dim=0).to(device)
    with torch.no_grad():
        output = net(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()
    
    fig = plt.figure(figsize=(5 * 2.5, 4 * 3), dpi=100)
    for j in range(20):
        ax = fig.add_subplot(4, 5, j+1, xticks=[], yticks=[])
        npimg = images[j].cpu().numpy().transpose(1, 2, 0)
        npimg *= 255
        plt.imshow(npimg.astype('uint8'))
        title = "{},{:.2f}%\n(label:{}".format(
            class_index[str(preds[j])],
            probs[j]*100,
            class_index[str(labels[j])]
        )
        ax.set_title(title, color=("green" if preds[j] == labels[j] else "red"))
    return fig