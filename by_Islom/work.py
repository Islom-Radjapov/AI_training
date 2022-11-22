import numpy as np
import matplotlib.pyplot as plt

# (Modelimiz) Tog'ri hisoblash uchun funtsiya
def forward(x):
    return x * w

# Xatolik (Loss) ning funtsiya
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

x_soat = [1.0, 2.0, 3.0]
y_baho = [4.0, 5.0, 6.0]

w_list = []
mse_list  = []

for w in np.arange(0.0, 4.1, 0.1):
    print("w={:.3f}".format(w))
    L_umum = 0

    for x_hb_qiym, y_hb_qiym in zip(x_soat, y_baho):
        y_hb_bash = forward(x_hb_qiym)
        L_hb_qiym = loss(x_hb_qiym, y_hb_qiym)
        L_umum += L_hb_qiym
        print("\t", "{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(x_hb_qiym, y_hb_qiym, y_hb_bash, L_hb_qiym))

    # Harbir malumot uchun MSE ni hisoblaymiz
    print("MSE=>", L_umum / len(x_soat))
    w_list.append(w)
    mse_list.append(L_umum / len(x_soat))

# Grafik natija
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()



# Training Data (O'rgatishdagi malumotlar)
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]

# w uchun dastlabgi qiymat
# w = 1.0

# (Modelimiz) Tog'ri hisoblash uchun funtsiya
def forward(x):
    return x * w

# Xatolik (Loss) ning funtsiya
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

# Gradient uchunfuntsiya
def gradient(x, y): # d_loss / d_w
    return 2 * x * (x * w - y)


# Traingdan avval
# print(forward(4))