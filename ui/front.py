import streamlit as st
import torch
from torchvision.transforms import ToTensor
from PIL import Image

torch.__version__

# 假设模型返回10个类别的预测结果
num_classes = 10

# 创建随机初始化的模型
model = torch.nn.Sequential(
    torch.nn.Linear(3 * 224 * 224, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, num_classes),
)

# 加载示例图片用于演示
sample_image = Image.open("sample_image.jpg")

st.title("Breast Cancer detection")
st.write("upload a image")

# 创建上传文件界面
uploaded_file = st.file_uploader("upload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # 将图像转换为张量，并进行模型预测
    image_tensor = ToTensor()(image).unsqueeze(0)
    flattened_image = image_tensor.view(-1, 3 * 224 * 224)
    prediction = model(flattened_image)
    _, predicted_class = torch.max(prediction, 1)

    st.write("预测类别：", predicted_class.item())

st.write("Images：")
st.image(sample_image, caption='Breast Cancer detection Image', use_column_width=True)