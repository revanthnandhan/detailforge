import streamlit as st
import cv2
import os
import tempfile
from realesrgan import RealESRGANer
import time
import random
import torch

# Due to different versions of torchvision, it may cause errors: https://github.com/xinntao/Real-ESRGAN/issues/859
import sys
import types
try:
    # Check if `torchvision.transforms.functional_tensor` and `rgb_to_grayscale` are missing
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    # Import `rgb_to_grayscale` from `functional` if it’s missing in `functional_tensor`
    from torchvision.transforms.functional import rgb_to_grayscale
    # Create a module for `torchvision.transforms.functional_tensor`
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    # Add this module to `sys.modules` so other imports can access it
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet


@st.cache_resource
def load_model(model_name, device="cpu", tile=0):
    model_configs = {
        'RealESRGAN_x4plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4),
        'RealESRNet_x4plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4),
        'RealESRGAN_x4plus_anime_6B': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4),
        'RealESRGAN_x2plus': (RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2)
    }

    if model_name not in model_configs:
        raise ValueError(f'Unsupported model name {model_name}')

    model, netscale = model_configs[model_name]
    model_path = os.path.join('weights', model_name + '.pth')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model file {model_path} not found, please download it first')
    
    print(f'Using model {model_name}')

    half = device != 'cpu'

    return RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device
    )

def main():
    # Output folder
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Clear all files in the output folder
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    
    st.title("Image Super-Resolution based on Real-ESRGAN")

    # --- START OF CHANGES ---
    
    # Define the mapping from display names to actual model names
    model_map = {
        'Upscaling x4 (General)': 'RealESRGAN_x4plus',
        'Upscaling x4 (Smoother)': 'RealESRNet_x4plus',
        'Upscaling x4 (Anime)': 'RealESRGAN_x4plus_anime_6B',
        'Upscaling x2 (General)': 'RealESRGAN_x2plus',
    }

    # Use the display names (keys of the dictionary) for the selectbox
    display_name = st.selectbox(
        "Select Model",
        list(model_map.keys()) 
    )
    
    # Get the actual model name from the selected display name
    model_name = model_map[display_name]

    # --- END OF CHANGES ---

    device_option = st.selectbox(
        "Select Device",
        ['cuda:0' if torch.cuda.is_available() else 'cpu', 'cpu']
    )
    tile = st.number_input("Tile Parameter (Split original image to reduce GPU memory usage, 0 means no split)", min_value=0, max_value=512, value=0, step=1)

    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = None

    if st.button('Load Model'):
        # Use the internal model_name for loading, but the display_name for the message
        st.session_state.model_handler = load_model(model_name, device=device_option, tile=tile)
        st.write(f"Model {display_name} has been loaded, Device: {device_option}, Tile: {tile}") # Show user-friendly name

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.read())
            input_image_path = temp_file.name

        img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        st.image(input_image_path, caption='Original Image', use_container_width=True)
        
        # Clean up the temp file immediately after reading
        if os.path.exists(input_image_path):
            os.remove(input_image_path)

        if img is not None:
            if st.button('Start Conversion'):
                if st.session_state.model_handler is None:
                    st.error("Please load the model first!")
                else:
                    with st.spinner('Converting, please wait...'):
                        # Get the actual scale from the loaded model
                        out_scale = st.session_state.model_handler.scale
                        output, _ = st.session_state.model_handler.enhance(img, outscale=out_scale)
                        
                        # Create filename based on time and random number
                        filename = f"{int(time.time())}_{random.randint(0, 1000)}.png"
                        output_image_path = os.path.join('output', filename)
                        cv2.imwrite(output_image_path, output)

                        st.image(output_image_path, caption='Converted Image', use_container_width=True)
                        # Provide download button
                        with open(output_image_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Image",
                                data=file,
                                file_name=filename,
                                mime="image/png"
                            )
        else:
            st.write("Unable to read the uploaded image!")

if __name__ == "__main__":
    main()