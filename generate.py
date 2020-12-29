# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import logging
import math
import os
import pickle
import re
import sys
import time

import cv2
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client


def is_perfect_cube(x):
    x = abs(x)
    return int(round(x ** (1. / 3))) ** 3 == x


def get_cube(x):
    return x ** (1. / 3)


def remap(value, from1, to1, from2, to2):
    return (value - from1) / (to1 - from1) * (to2 - from2) + from2


def remap2(value, low1, low2, high1, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)


# ----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, truncation_psi, outdir, class_idx, dlatents_npz):
    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # # output file name, encoder, fps, size(fit to image size)
    # video_writer = cv2.VideoWriter('generate_gan.mp4', fourcc, 20.0, (128, 128))

    # if not video_writer.isOpened():
    #     print("can't be opened")
    #     sys.exit()

    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:
        print(f'Generating images from dlatents file "{dlatents_npz}"')
        dlatents = np.load(dlatents_npz)['dlatents']
        assert dlatents.shape[1:] == (18, 512)  # [N, 18, 512]
        imgs = Gs.components.synthesis.run(dlatents,
                                           output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        for i, img in enumerate(imgs):
            fname = f'{outdir}/dlatent{i:02d}.png'
            print(f'Saved {fname}')
            PIL.Image.fromarray(img, 'RGB').save(fname)
        return

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    rnd = np.random.RandomState(600)
    z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]

    def pil2cv(image):
        ''' PIL型 -> OpenCV型 '''
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image

    # cap = cv2.VideoCapture("out.mp4")
    # if not cap.isOpened():
    #     exit(0)

    z[0] = [0.0 for val in range(512)]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    images = Gs.run(z, label, **Gs_kwargs)  # [minibatch, height, width, channel]

    # isGotMessage = False
    # seedsValues = []
    # def messege_handler(unused_addr, *p):
    #     try:
    #         print(p)
    #
    #         # tuple to list
    #         z[0] = [val for val in p]
    #
    #         print(Gs)
    #
    #         # generate image
    #         tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    #         images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    #         pil_image = PIL.Image.fromarray(images[0], 'RGB')
    #         pil_image.save("generate.jpg")
    #         print(f"save generate.jpg ")
    #     except ValueError:
    #         pass
    3#
    # OSC_dispatcher = dispatcher.Dispatcher()
    # OSC_dispatcher.map("/generate", messege_handler)
    # server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 12000), OSC_dispatcher)
    # server.serve_forever()

    while True:

        try:
            data = []
            f = open("request.txt")
            for val in f.read().split(","):
                if val is not "":
                    data.append(float(val))
            f.close()

            z[0] = data
        except ValueError:
            continue

        # generate image
        print("------------------------------------------------------")
        print(data)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
        pil_image = PIL.Image.fromarray(images[0], 'RGB')
        try:
            pil_image.save("generate.jpg")
        except PermissionError:
            logging.error("can not generate generate.jpg, Permission Error")
        # print(f"save generate.jpg ")

        time.sleep(1.0 / 10)

    # counter = 0
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         new_z = []
    #         for rgb in frame:
    #             new_val = (rgb[0][0]+1) * (rgb[0][1]+1) * (rgb[0][2]+1)
    #             max = 256**3.0
    #             # after_range = 0.000005
    #             after_range = 1.0
    #             new_val = remap2(new_val, 0, max, -after_range, after_range)
    #             new_z.append(-new_val)
    #         z[0] = new_z
    #         tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    #         images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    #         pil_image = PIL.Image.fromarray(images[0], 'RGB')
    #         print(f"writing image: {outdir}/{counter}.png")
    #         counter = counter + 1
    #         cv2_image = pil2cv(pil_image)
    #         video_writer.write(cv2_image)
    #     else:
    #         cap.release()
    #         video_writer.release()
    #         break


# ----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''


# ----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    g.add_argument('--dlatents', dest='dlatents_npz', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)',
                        default=0.5)
    parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

    args = parser.parse_args()

    generate_images(**vars(args))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
