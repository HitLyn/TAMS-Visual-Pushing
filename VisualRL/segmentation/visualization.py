import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from VisualRL.vae.model import VAE
import torch
from torchvision import transforms
from IPython import embed
import matplotlib.pyplot as plt

IMAGE_PATH = "/homeL/cong/HitLyn/Visual-Pushing/images/all_objects_masks_random"
SAVE_PATH = "/homeL/cong/HitLyn/Visual-Pushing/images/save_from_visualization_new"
SCALE_RANGE = 6
CHOSEN_FEATURES = 10

device = torch.device('cuda:1')
# global model
model = VAE(device = device, image_channels = 1, h_dim = 1024, z_dim = 6)
model.load("/homeL/cong/HitLyn/Visual-Pushing/results/vae/6_new/vae_model", 100, map_location='cuda:1')

random_id = 1


window = tk.Tk()
window.title("VAE Latent Space")

frame = tk.Frame(master = window, width = 1200, height = 450)
frame.pack()

l1 = tk.Label(master = frame, text = '')
l1.config(font=("Courier", 14))
l1.place(x = 60, y = 330)

l2 = tk.Label(master = frame, text = '')
l2.config(font=("Courier", 14))
l2.place(x = 880, y = 330)

l3 = tk.Label(master = frame, text = 'latent features')
l3.config(font=("Courier", 14))
l3.place(x = 490, y = 10)

encoder_image = Image.open("/homeL/cong/Downloads/encoder.png").resize([100, 150])
encoder_image = ImageTk.PhotoImage(encoder_image)
l4 = tk.Label(master = frame)
l4.config(image=encoder_image)
l4.place(x = 350, y = 120)
decoder_image = Image.open("/homeL/cong/Downloads/decoder.png").resize([100, 150])
decoder_image = ImageTk.PhotoImage(decoder_image)
l5 = tk.Label(master = frame)
l5.config(image = decoder_image)
l5.place(x = 700, y = 120)

label_input = tk.Label(master = frame)
label_input.place(x = 50, y = 50)

label_output = tk.Label(master = frame)
label_output.place(x = 850, y = 50)

def set_s1_value(val):
    # l1.configure(text = val)
    # print("latent_value", latent_value)
    # mu = latent_value.copy()
    mu = latent_value
    mu[0] = val
    show_out_put(mu)
    # print(latent_value)

def set_s2_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[1] = val
    show_out_put(mu)

def set_s3_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[2] = val
    show_out_put(mu)

def set_s4_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[3] = val
    show_out_put(mu)

def set_s5_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    # embed()
    mu[4] = val
    show_out_put(mu)

def set_s6_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[5] = val
    show_out_put(mu)

def set_s7_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[6] = val
    show_out_put(mu)

def set_s8_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[7] = val
    show_out_put(mu)

def set_s9_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[8] = val
    show_out_put(mu)

def set_s10_value(val):
    # l1.configure(text = val)
    # mu = latent_value.copy()
    mu = latent_value
    mu[9] = val
    show_out_put(mu)

def show_out_put(mu):
    # print(mu)
    with torch.no_grad():
        image_recon_tensor = model.decode(torch.tensor(mu).to(dtype = torch.float32).to(device).unsqueeze(0))
    image_recon_pil = transforms.ToPILImage()(image_recon_tensor.squeeze().cpu())

    # image_recon_tk = ImageTk.PhotoImage(image_recon_pil)
    image_recon_tk = ImageTk.PhotoImage(image_recon_pil.resize([256, 256]))
    label_output.configure(image = image_recon_tk)
    label_output.photo = image_recon_tk
    # print('update output image...')


def show_image(id):
    global latent_value
    file_name = os.path.join(IMAGE_PATH, "{:0>5d}.png".format(id))
    l1.configure(text = "input image: {:0>5d}.png".format(id))
    l2.configure(text="reconstruct image".format(id))
    image = Image.open(file_name)
    # embed();exit()
    image_tk = ImageTk.PhotoImage(image.resize([256,256]))
    # image_tk = ImageTk.PhotoImage(image)
    label_input.configure(image = image_tk)
    label_input.photo = image_tk
    # reset slider
    tensor_image = transforms.ToTensor()(transforms.Resize(64)(image)).unsqueeze(0).to(device)
    # embed()
    with torch.no_grad():
        image_recon, z, mu, logvar = model(tensor_image)
        image_recon = model.predict(tensor_image)
    latent_value = mu[0].cpu().numpy()
    # print(latent_value)
    show_out_put(latent_value)

    # reset values
    s1.set(latent_value[0])
    show_out_put(latent_value)
    s2.set(latent_value[1])
    show_out_put(latent_value)
    if len(latent_value) > 2:
        s3.set(latent_value[2])
        show_out_put(latent_value)
    if len(latent_value) >3:
        s4.set(latent_value[3])
        show_out_put(latent_value)
    if len(latent_value) > 4:
        s5.set(latent_value[4])
        show_out_put(latent_value)
    if len(latent_value) > 5:
        s6.set(latent_value[5])
        show_out_put(latent_value)
    if len(latent_value) > 6:
        s7.set(latent_value[6])
        show_out_put(latent_value)
    if len(latent_value) > 7:
        s8.set(latent_value[7])
        show_out_put(latent_value)
    if len(latent_value) > 8:
        s9.set(latent_value[8])
        show_out_put(latent_value)
    if len(latent_value) > 9:
        s10.set(latent_value[9])
        show_out_put(latent_value)

def select_image_from_train():
    # print('selecting new')
    global random_id
    # random_id = np.random.randint(18000, 19000)
    random_id = np.random.randint(0, 23000)
    # random_id = 1
    # num = 1
    show_image(random_id)

def select_image_from_test():
    # print('selecting new')
    global random_id
    random_id = np.random.randint(25000, 29000)
    # random_id = np.random.randint(0, 18000)
    # random_id = 18005
    show_image(random_id)

def next_image():
    global random_id
    random_id += 1
    show_image(random_id)

def last_image():
    global random_id
    random_id -= 1
    show_image(random_id)

def save_image():
    global latent_value
    global random_id
    input_file_name = os.path.join(IMAGE_PATH, "{:0>5d}.png".format(random_id))
    input_image = Image.open(input_file_name)
    save_file_name = os.path.join(SAVE_PATH, "{:0>5d}_input.png".format(random_id))
    # plt.imsave(save_file_name, input_image, format = 'png')
    cv2.imwrite(save_file_name, np.array(input_image))

    with torch.no_grad():
        image_recon_tensor = model.decode(torch.tensor(latent_value).to(dtype = torch.float32).to(device).unsqueeze(0))
    image_recon_pil = transforms.ToPILImage()(image_recon_tensor.squeeze().cpu())
    output_file_name = os.path.join(SAVE_PATH, "{:0>5d}_output.png".format(random_id))
    # plt.imsave(output_file_name, image_recon_pil, format = 'png')
    cv2.imwrite(output_file_name, np.array(image_recon_pil))


# slider
s1 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s1_value, width = 8)
s1.place(x = 500, y = 40)
s2 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s2_value, width = 8)
s2.place(x = 500, y = 70)
s3 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s3_value, width = 8)
s3.place(x = 500, y = 100)
s4 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s4_value, width = 8)
s4.place(x = 500, y = 130)
s5 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s5_value, width = 8)
s5.place(x = 500, y = 160)
s6 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s6_value, width = 8)
s6.place(x = 500, y = 190)
s7 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s7_value, width = 8)
s7.place(x = 500, y = 220)
s8 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s8_value, width = 8)
s8.place(x = 500, y = 250)
s9 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s9_value, width = 8)
s9.place(x = 500, y = 280)
s10 = tk.Scale(window, from_ = -SCALE_RANGE, to= SCALE_RANGE, orient=tk.HORIZONTAL, length = 150, resolution = 0.01, command = set_s10_value, width = 8)
s10.place(x = 500, y = 310)
# button
button = tk.Button(frame, text = "train set", command = select_image_from_train, font=("Courier", 14))
button.place(x = 80, y = 370)
button = tk.Button(frame, text = "test set", command = select_image_from_test, font=("Courier", 14))
button.place(x = 80, y = 410)
button = tk.Button(frame, text = "next", command = next_image, font=("Courier", 14))
button.place(x = 220, y = 370)
button = tk.Button(frame, text = "last", command = last_image, font=("Courier", 14))
button.place(x = 220, y = 410)
button = tk.Button(frame, text = "save image", command = save_image, font=("Courier", 14))
button.place(x = 500, y = 390)
window.mainloop()
