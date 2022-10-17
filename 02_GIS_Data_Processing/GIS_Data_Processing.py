'''
    _________________________________
    |                               |
    |   Tên: Huỳnh Viết Tuấn Kiệt   |
    |   MSSV: 20521494              |
    |_______________________________|
'''

import geopandas as gp
import pandas as pd
import numpy as np

path = "./CSL_HCMC/Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp"
file = open(path)
df = gp.read_file(path)
df_data = pd.DataFrame(df)

print("=========================================================================================")
q1 = df_data.to_numpy()[df_data['Shape_Area'].idxmax()]
print(f"Phường có diện tích lớn nhất: phường {q1[0]}, quận {q1[1]} ")
print(f"Diện tích: {q1[10]}")

print("=========================================================================================")
q2 = df_data.to_numpy()[df_data['Pop_2019'].idxmax()]
print(f"Phường có dân số 2019 cao nhất: phường {q2[0]}, quận {q2[1]} ")
print(f"Dân số: {q2[6]} người")

print("=========================================================================================")
q3 = df_data.to_numpy()[df_data['Shape_Area'].idxmin()]
print(f"Phường có diện tích nhỏ nhất: phường {q3[0]}, quận {q3[1]} ")
print(f"Diện tích: {q3[10]}")

print("=========================================================================================")
q4 = df_data.to_numpy()[df_data['Pop_2019'].idxmin()]
print(f"Phường có dân số 2019 thấp nhất: phường {q4[0]}, quận {q4[1]} ")
print("Dân số:", q4[6], "người")

S = df_data['Pop_2019'].to_numpy()
A = df_data['Pop_2009'].to_numpy()
N = 2019 - 2009 + 1
'''
    Formula: er = S / (A * N)
        + er: Tốc độ gia tăng dân số
        + S: N năm sau A năm
        + A: Năm làm mốc
        + N: (S - A) năm
'''

print("=========================================================================================")
er5 = (S * 100) / (A * N)
q5_index = np.argmax(er5)
q5 = df_data.to_numpy()[q5_index]
print(
    f"Phường có tốc độ tăng trưởng dân số nhanh nhất (dựa trên Pop_2009 và Pop_2019): phường {q5[0]}, quận {q5[1]} ")
print(f"Tốc độ tăng trưởng: {round(er5[q5_index], 3)}%")

print("=========================================================================================")
er6 = (S * 100) / (A * N)
q6_index = np.argmin(er6)
q6 = df_data.to_numpy()[q6_index]
print(
    f"Phường có tốc độ tăng trưởng dân số thấp nhất (dựa trên Pop_2009 và Pop_2019): phường {q6[0]}, quận {q6[1]} ")
print(f"Tốc độ tăng trưởng dân số: {round(er6[q6_index], 3)}%")

print("=========================================================================================")
b7 = np.abs(S - A)
q7_index = np.argmax(b7)
q7 = df_data.to_numpy()[q7_index]
print(f"Phường có biến động dân số nhanh nhất: Phường {q7[0]}, quận {q7[1]}")
print(f"Mức biến động: {b7[q7_index]} dân")

print("=========================================================================================")
b8 = np.abs(S - A)
q8_index = np.argmin(b8)
q8 = df_data.to_numpy()[q8_index]
print(f"Phường có biến động dân số chậm nhất: Phường {q8[0]}, quận {q8[1]}")
print(f"Mức biến động: {b8[q8_index]} dân")

print("=========================================================================================")
q9 = df_data.to_numpy()[df_data['Den_2019'].idxmax()]
print(f"Phường có mật độ dân số cao nhất: Phường {q9[0]}, quận {q9[1]}")
print(f"Mật độ: {q9[8]} người/km2")

print("=========================================================================================")
q10 = df_data.to_numpy()[df_data['Den_2019'].idxmin()]
print(f"Phường có mật độ dân số cao nhất: Phường {q10[0]}, quận {q10[1]}")
print(f"Mật độ: {q10[8]} người/km2")
