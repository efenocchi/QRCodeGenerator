import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()

from nodes import (
    KSampler,
    CLIPTextEncode,
    ControlNetApplyAdvanced,
    VAEDecode,
    CheckpointLoaderSimple,
    LoadImage,
    ControlNetLoader,
    NODE_CLASS_MAPPINGS,
    EmptyLatentImage,
    VAELoader,
)


def main():
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="dreamshaper_8.safetensors"
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=768, height=768, batch_size=4
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="a cyborg\n", clip=get_value_at_index(checkpointloadersimple_4, 1)
        )

        cliptextencode_7 = cliptextencode.encode(
            text="ugly, artefacts, bad",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        controlnetloader = ControlNetLoader()
        controlnetloader_10 = controlnetloader.load_controlnet(
            control_net_name="control_v11f1e_sd15_tile.pth"
        )

        controlnetloader_11 = controlnetloader.load_controlnet(
            control_net_name="control_v1p_sd15_brightness.safetensors"
        )

        loadimage = LoadImage()
        loadimage_14 = loadimage.load_image(image="qr-code_resized.png")

        vaeloader = VAELoader()
        vaeloader_24 = vaeloader.load_vae(
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
        )

        controlnetapplyadvanced = ControlNetApplyAdvanced()
        ksampler = KSampler()
        vaedecode = VAEDecode()

        for q in range(10):
            controlnetapplyadvanced_28 = controlnetapplyadvanced.apply_controlnet(
                strength=0.75,
                start_percent=0.35000000000000003,
                end_percent=0.6,
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                control_net=get_value_at_index(controlnetloader_10, 0),
                image=get_value_at_index(loadimage_14, 0),
            )

            controlnetapplyadvanced_27 = controlnetapplyadvanced.apply_controlnet(
                strength=0.35000000000000003,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(controlnetapplyadvanced_28, 0),
                negative=get_value_at_index(controlnetapplyadvanced_28, 1),
                control_net=get_value_at_index(controlnetloader_11, 0),
                image=get_value_at_index(loadimage_14, 0),
            )

            ksampler_17 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                positive=get_value_at_index(controlnetapplyadvanced_27, 0),
                negative=get_value_at_index(controlnetapplyadvanced_27, 1),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_18 = vaedecode.decode(
                samples=get_value_at_index(ksampler_17, 0),
                vae=get_value_at_index(vaeloader_24, 0),
            )


if __name__ == "__main__":
    main()
