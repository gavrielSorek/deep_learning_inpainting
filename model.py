import random

from torch import nn, optim
import torch
import matplotlib.pyplot as plt
import torchvision
import os
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from torchvision import transforms, models
import os.path

# init
colab_suffix = ""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
size_of_picture = 224
model = models.alexnet(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False

# init layers for style calculation
layers_style = {
    0: 'encoder_conv1',
    3: 'encoder_conv2',
    6: 'encoder_conv3',
    8: 'encoder_conv4',
    10: 'encoder_conv5',
}

style_weights = {
    "encoder_conv1": 0.8,
    "encoder_conv2": 0.1,
    "encoder_conv3": 0.4,
    "encoder_conv4": 0.2,
    "encoder_conv5": 0.2
}

# init losses
mse_loss = nn.MSELoss()
criterion = nn.BCELoss()

# define constants
train_batch_size = 20
validation_batch_size = 20
batch_size = 20
max_pixel_val = 255
min_pixel_val = 0

resize_to_224 = transforms.Resize(224)
resize_to_256 = transforms.Resize(256)

# transforms
preprocess = nn.Sequential(
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)
invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.]),
                               ])


# return mask from image path
def get_mask_from_image(image_path):
    mask = resize_to_224(torchvision.io.read_image(image_path).float())
    mask = get_zero_one_mask(mask)
    return mask[0]


center_mask = None
arbitrary_mask = None


# return random masks
def get_random_blocks_mask():
    global center_mask, arbitrary_mask
    if center_mask is None or arbitrary_mask is None:
        center_mask = get_mask_from_image("./" + colab_suffix + 'Masks/center.jpg')
        arbitrary_mask = get_mask_from_image("./" + colab_suffix + 'Masks/arbitrary.jpg')
    rand = random.randint(0, 2)
    if rand == 1:
        return center_mask
    if rand == 2:
        return arbitrary_mask
    imageSize = 224
    blocksize = 40
    numerOfBlocks = random.randint(5, 9)
    startPoints = []
    for i in range(numerOfBlocks):
        x = random.randint(0, imageSize - blocksize)
        y = random.randint(0, imageSize - blocksize)
        startPoints.insert(1, [x, y])

    M = torch.zeros(3, 224, 224, dtype=torch.uint8)
    for point in startPoints:
        for k1 in range(point[0], point[0] + blocksize):
            for k2 in range(point[1], point[1] + blocksize):
                M[0][k1][k2] = 1
                M[1][k1][k2] = 1
                M[2][k1][k2] = 1
    return M[0]


def normalize(tensor):
    tensor = tensor / 255
    tensor = preprocess(tensor)
    return tensor


def unnormalize(tensor):
    tensor = invTrans(tensor)
    tensor *= 255
    return tensor


def fill_tensor_with_dir_content(tensor, directory, limit):
    files = [name for name in os.listdir(directory)]
    for i, filename in enumerate(files):
        if i >= limit:
            break
        tensor[i] = normalize(torchvision.io.read_image(os.path.join(directory, filename)).float())


def get_files_amount_in_directory(directory):
    num = 0
    for name in os.listdir(directory):
        num += 1
    return num


# get train and validation loaders
def get_data_loaders(train_path, validation_path):
    train_size = get_files_amount_in_directory(train_path)
    validation_size = get_files_amount_in_directory(validation_path)
    train_tensor = torch.zeros(train_size, 3, size_of_picture, size_of_picture, dtype=torch.float)
    validation_tensor = torch.zeros(validation_size, 3, size_of_picture, size_of_picture, dtype=torch.float)
    fill_tensor_with_dir_content(train_tensor, train_path, train_size)
    fill_tensor_with_dir_content(validation_tensor, validation_path, validation_size)
    train_loader, validation_loader = get_train_validation_loaders(train_tensor, validation_tensor)
    return train_loader, validation_loader


# show image from tensor
def show_image(tensor_image, title):
    tensor_image = unnormalize(tensor_image)
    tensor_image = tensor_image.permute(1, 2, 0)
    plt.imshow(tensor_image.int().cpu())
    plt.title(title)
    plt.show()


# return x,y tensor form data tensor
def get_x_y_from_data_tensor(data_tensor):
    train_tensor_x = torch.clone(data_tensor)
    train_tensor_y = torch.zeros(data_tensor.size(dim=0), 4, 224, 224, dtype=torch.float)
    for i in range(data_tensor.size(dim=0)):
        M = get_random_blocks_mask()
        oneMatrix = torch.full((size_of_picture, size_of_picture), 1)
        Mask = oneMatrix - M
        train_tensor_x[i] = (data_tensor[i] * Mask)
        train_tensor_y[i, :3, :, :] = data_tensor[i]
        train_tensor_y[i, 3, :, :] = M
    return train_tensor_x, train_tensor_y


# return loader
def get_loader_by_x_y_tensors(train_tensor_x, train_tensor_y, batch_size):
    train_data_set = TensorDataset(train_tensor_x, train_tensor_y)  # create your datset
    return torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)


# return train and validation loaders
def get_train_validation_loaders(train_tensor, validation_tensor):
    train_tensor_x, train_tensor_y = get_x_y_from_data_tensor(train_tensor)
    validation_x, validation_y = get_x_y_from_data_tensor(validation_tensor)
    return get_loader_by_x_y_tensors(train_tensor_x, train_tensor_y, train_batch_size), get_loader_by_x_y_tensors(
        validation_x,
        validation_y,
        validation_batch_size)


# implement the encoder decoder channel wise
class EncoderToDecoder(nn.Module):
    def __init__(self, m, n):
        super(EncoderToDecoder, self).__init__()
        self.drop = nn.Dropout(0.4)
        self.num_of_neurons = m * n * n
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm1d(self.num_of_neurons)
        self.m = m
        self.n = n
        self.out_channels_encoder = torch.randn(self.m, self.n * self.n, self.n * self.n, requires_grad=True).to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.drop(x)
        x = x.view(size=(-1, self.m, self.n * self.n))
        x = x.permute(1, 0, 2)
        x = torch.matmul(x, self.out_channels_encoder.to(device)).squeeze(-1).to(device)
        x = x.permute(1, 0, 2)
        x = torch.reshape(x, shape=(-1, self.num_of_neurons))
        x = self.bn(x)
        x = self.prelu(x)
        x = x.view(size=(-1, self.m, self.n, self.n))
        return x


# do nothing
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.m = 256  # num of encoder out channels
        self.n = 6  # width/ height
        self.encoder_to_decoder = nn.Sequential(
            EncoderToDecoder(self.m, self.n)
        ).to(device)
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ).to(device)
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ).to(device)
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ).to(device)
        self.decoder_conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ).to(device)
        self.decoder_conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        ).to(device)

    def forward(self, x):
        x = x.view(size=(-1, 256, self.n, self.n))
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_conv3(x)
        x = self.decoder_conv4(x)
        x = self.decoder_conv5(x)
        x = resize_to_224(x)
        return x


# Discriminator Net
class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.drop1 = nn.Dropout(0.3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ).to(device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ).to(device)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ).to(device)
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1
            ),
            nn.Sigmoid()
        ).to(device)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


# return discriminator loss
def discriminator_loss(output, labels, discriminator_model, batch_size):
    discriminator_output = discriminator_model(output)
    combine_real_fake_x = torch.cat((discriminator_model(labels), discriminator_output), 0).to(device)
    # create the data for classifier
    dis_output_size = discriminator_output.size(dim=2)
    y_real = torch.full((batch_size, 1, dis_output_size, dis_output_size), 1, dtype=torch.float)
    y_fake = torch.full((batch_size, 1, dis_output_size, dis_output_size), 0, dtype=torch.float)

    combine_real_fake_y = torch.cat((y_real, y_fake), 0).to(device)

    # shuffle data
    data_size = batch_size
    order = np.array(range(data_size))
    np.random.shuffle(order)
    combine_real_fake_x[np.array(range(data_size))] = combine_real_fake_x[order]
    combine_real_fake_y[np.array(range(data_size))] = combine_real_fake_y[order]
    # return loss
    return criterion(combine_real_fake_x, combine_real_fake_y).to(device)


def get_masked_tensor(tensor, mask):
    temp_tensor = torch.zeros(tensor.size(dim=0), 3, size_of_picture, size_of_picture, dtype=torch.float).to(device)
    # for all RGB we applying the mask
    temp_tensor[:, 0, :, :] = tensor[:, 0, :, :].to(device) * mask.to(device)
    temp_tensor[:, 1, :, :] = tensor[:, 1, :, :].to(device) * mask.to(device)
    temp_tensor[:, 2, :, :] = tensor[:, 2, :, :].to(device) * mask.to(device)
    return temp_tensor


def context_encoder_loss(output, labels, discriminator_model, is_monet):
    reconstruction_loss = mse_loss(output.to(device), labels.to(device))
    # flip labels and output
    discriminator_random_loss = discriminator_loss(output=labels.to(device),
                                                   labels=output,
                                                   discriminator_model=discriminator_model,
                                                   batch_size=train_batch_size)
    if is_monet:
        return 0.999 * reconstruction_loss + 0.001 * discriminator_random_loss + 0.01 * calculate_style_loss(output,
                                                                                                             labels)
    else:
        return 0.999 * reconstruction_loss + 0.001 * discriminator_random_loss


# remove model
def remove_model(model_path):
    if os.path.isfile(model_path):
        os.remove(model_path)


def train(epoch, model, optimizer, discriminator_model, discriminator_optimizer, is_monet, train_loader,
          validation_loader):
    model.train()
    min_validation_loss = 100000  # like infinity
    for i in range(epoch):
        iter_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            discriminator_model.zero_grad()
            dis_loss = discriminator_loss(model(data.to(device)), labels[:, :3, :, :].to(device),
                                          discriminator_model.to(device), train_batch_size)
            dis_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator_model.parameters(), 1)
            discriminator_optimizer.step()

            model.zero_grad()
            output = model(data.to(device))

            random_block_loss = context_encoder_loss(output, labels[:, :3, :, :], discriminator_model,
                                                     is_monet)
            iter_loss += random_block_loss
            random_block_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        # # save best model
        current_validation_loss = get_loss(model_local=model,
                                           discriminator_model=discriminator_model,
                                           is_monet=is_monet,
                                           data_loader=validation_loader)
        if min_validation_loss > current_validation_loss:
            if is_monet:
                remove_model("./" + colab_suffix + "client_saved_model/best_model_monet")
                torch.save(model, "./" + colab_suffix + "client_saved_model/best_model_monet")
            else:
                remove_model("./" + colab_suffix + "client_saved_model/best_model")
                torch.save(model, "./" + colab_suffix + "client_saved_model/best_model")
            min_validation_loss = current_validation_loss
        print("************************************************************************************")
        print("                         epoch " + str(i) + " /" + str(epoch - 1))
        print("train loss " + str(
            get_loss(model, discriminator_model, is_monet, train_loader)))
        print("validation loss " + str(
            get_loss(model, discriminator_model, is_monet, validation_loader)))
        print("************************************************************************************")


def compose_pic_with_random_blocks(pic_with_hols, output, mask):
    new_pic = torch.clone(pic_with_hols)
    new_pic = new_pic * (1 - mask)
    new_pic = new_pic.to(device) + output.to(device) * mask.to(device)
    return new_pic


def apply_mask(pic, mask):
    pic = pic.to(device)
    mask = mask.to(device)
    return pic * (1 - mask) + mask


# return the loss on validation
def get_loss(model_local, discriminator_model, is_monet, data_loader):
    iter_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model_local(data.to(device))
            loss = context_encoder_loss(output, target[:, :3, :, :], discriminator_model, is_monet)
            iter_loss += loss
    validation_loss = iter_loss / (len(data_loader.dataset) / batch_size)
    model_local.zero_grad()
    discriminator_model.zero_grad()
    return validation_loss


# return zero one mask
def get_zero_one_mask(mask_tensor):
    for k in range(3):
        for i in range(mask_tensor.size(dim=1)):
            for j in range(mask_tensor.size(dim=2)):
                if (mask_tensor[k][i][j] != 0):
                    mask_tensor[k][i][j] = 1
    return mask_tensor


# predict and save the predictions
def predict_and_save_pic(model_local, mask, pic):
    model_local.eval()
    pic = torch.unsqueeze(pic, 0)
    prediction = model_local(pic * (1 - mask))
    composed_pic = compose_pic_with_random_blocks(pic[0], prediction[0], mask[0])
    composed_pic = unnormalize(composed_pic)
    composed_pic = resize_to_256(composed_pic)
    rand = random.randint(0, 9999999)
    print("your image path is " + './' + colab_suffix + 'predictions/image_' + str(rand) + 'predicted.png')
    save_image(composed_pic / 255,
               './' + colab_suffix + 'predictions/image_' + str(rand) + 'predicted.png')


def run_trained_model_client(image_path, mask_path, model_to_run):
    print("waiting for results...")
    image_tensor = normalize(torchvision.io.read_image(os.path.join(image_path)).float())
    mask_tensor = get_zero_one_mask(resize_to_224(torchvision.io.read_image(os.path.join(mask_path))))
    predict_and_save_pic(model_to_run, mask_tensor, image_tensor)
    print("finished")


def train_model_client(train_folder_path, validation_folder_path, is_monet):
    print("training...")
    best_model = model
    discriminator_model = DiscriminatorNet().to(device)
    best_optim = optim.Adam(best_model.parameters(), lr=0.0006, betas=(0.5, 0.999))
    discriminator_optimizer = optim.SGD(discriminator_model.parameters(), lr=0.001)
    train_loader, validation_loader = get_data_loaders(train_folder_path, validation_folder_path)
    train(epoch=50, model=best_model, optimizer=best_optim, discriminator_model=discriminator_model,
          discriminator_optimizer=discriminator_optimizer, is_monet=is_monet,
          train_loader=train_loader, validation_loader=validation_loader)
    print("finished training")


########################################### STYLE ###########################################
def get_style_results_from_model(image):
    features_sytle = {}
    feature_layer = model.features
    for i in range(len(feature_layer)):
        image = feature_layer[i](image)
        if i in layers_style:
            features_sytle[layers_style[i]] = image
    return features_sytle


# return gram matrix
def gram_matrix(activation_image):
    m, c, w, h = activation_image.shape
    G = activation_image.view(m * c, w * h)
    gram_mat = torch.mm(G, G.t())
    return gram_mat


def style_loss(image_activation, target_activation):
    m, c, w, h = image_activation.shape
    image_gram = gram_matrix(image_activation)
    target_gram = gram_matrix(target_activation)
    return torch.sum((image_gram - target_gram) ** 2) / (4 * c * c * (w * h) ** 2)


# return style loss
def calculate_style_loss(image, target):
    style_loss_local = torch.zeros(1).to(device)
    style_loss_local.requires_grad = True
    image_activation_style = get_style_results_from_model(image.to(device))
    target_activation_style = get_style_results_from_model(target.to(device))
    for layer in style_weights:
        res = style_weights[layer] * style_loss(image_activation_style[layer], target_activation_style[layer])
        style_loss_local = style_loss_local + res.to(device)
    return style_loss_local


######################### START PROGRAM #########################
# main logic
def main():
    while True:
        option = input("Enter 1 for run the model with trained weights \n"
                       + "Enter 2 to train model \n"
                       + "Enter other key to exit \n")

        if option == '1':
            mask_path = input("Enter mask path: ")
            image_path = input("Enter image path: ")
            is_custom_model = input("if custom model press y for default model press another key: ")
            if is_custom_model == 'y':
                model_path = input("Enter model path: ")
                model_to_run = torch.load(model_path, map_location=torch.device('cpu'))
            else:
                is_monet = input("Enter 1 if monet, 0 otherwise: ")
                if is_monet == '1':
                    model_to_run = torch.load("./" + colab_suffix + "default_models/best_model_monet",
                                              map_location=torch.device('cpu'))
                else:
                    model_to_run = torch.load("./" + colab_suffix + "default_models/best_model",
                                              map_location=torch.device('cpu'))
            run_trained_model_client(image_path, mask_path, model_to_run)
        elif option == '2':
            is_monet = input("Enter 1 if monet, 0 otherwise: ")
            if is_monet == '1':
                is_monet = True
            else:
                is_monet = False
            train_folder_path = input("Enter folder path for training: ")
            validation_folder_path = input("Enter folder path for validation: ")
            train_model_client(train_folder_path, validation_folder_path, is_monet)
        else:
            return


def start_point():
    global model
    model.to(device)
    model.avgpool = Identity()
    model.classifier = Decoder()
    main()


start_point()
