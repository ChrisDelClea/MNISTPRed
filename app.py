import torch
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from train import Net
from scipy.ndimage.interpolation import zoom
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def main():
    st.title("MNIST Number Prediction")
    left_column, right_column = st.beta_columns(2)
    PATH = "./mnist_cnn.pt"
    model = Net()
    # model = torch.load(PATH)
    model.load_state_dict(torch.load(PATH))
    # model.eval()
    # st.write(model.eval())

    # Create a canvas component
    with left_column:
        st.header("Draw a number")
        st.subheader("[0-9]")
        canvas_result = st_canvas(
                fill_color="rgb(0, 0, 0)",  # Fixed fill color with some opacity
                # stroke_width="1, 25, 3",
                stroke_width = 10,
                stroke_color="#FFFFFF",
                background_color="#000000",
                update_streamlit=True,
                width=224,
                height=224,
                drawing_mode="freedraw",
                key="canvas",
        )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        grey = rgb2gray(img)
        grey = zoom(grey, 0.125)
        x_np = torch.from_numpy(grey).unsqueeze(0) #
        x = x_np.unsqueeze(0)
        x = x.float()
        output = model(x)
        pred = torch.max(output, 1)
        pred = pred[1].numpy()
    with right_column:
        st.header("Predicted Result")
        st.title(str(pred[0]))

if __name__ == '__main__':
    main()


