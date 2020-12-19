import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

url_ffhq = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir="cache") as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]


# ----------------  Style mixing -------------------

def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    print("-----------------------------------------")
    print(Gs.input_shape)
    print("-----------------------------------------")

    d1 = np.random.RandomState(512).randn(Gs.input_shape[1])  # seed = 503 のベクトルを取得
    d2 = np.random.RandomState(512).randn(Gs.input_shape[1])  # seed = 888 のベクトルを取得

    dx = (d2 - d1) / 3  # ３分割で補間
    steps = np.linspace(0, 3, 4)  # stepsに[0,1,2,3] を代入
    dst_latents = np.stack((d1 + dx * step) for step in steps)  # dst_latents にベクトルを４つスタック

    src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None)  # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * 5), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))

    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))

    png_filename = os.path.join("result", 'style_mix.png')
    canvas.save(png_filename)


# --------------- main -----------------

def main():
    tflib.init_tf()
    path = "test.pkl"
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    draw_style_mixing_figure(os.path.join(result_dir, 'style_mix.png'), load_Gs("test.pkl"), w=1024, h=1024,
                             src_seeds=[11, 701, 583], dst_seeds=[888, 829, 1898, 1733, 1614, 845],
                             style_ranges=[range(0, 4)] * 4)  # style_mixingのレンジ指定


if __name__ == "__main__":
    main()