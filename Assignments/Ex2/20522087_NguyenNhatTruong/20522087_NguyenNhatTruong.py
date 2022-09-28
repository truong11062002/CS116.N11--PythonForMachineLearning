import matplotlib.pyplot as plt
import geopandas as gpd

# print(gpd.__version__)

hcm_data = gpd.read_file("C:\CS116.N11\Assignments\Ex2\CSL_HCMC\Data\GIS\Population\population_HCMC\population_shapefile\Population_Ward_Level.shp")

# print(hcm_data.head())

# Vẽ biểu đồ theo tên các phường

# hcm_data.plot(column='Com_Name', figsize=(10,5))
# plt.show()

# Phường nào có diện tích lớn nhất
max_area= hcm_data['Shape_Area'].idxmax()
print("Phường có diện tích lớn nhất là phường: ", hcm_data.loc[max_area,'Com_Name'], "quận", hcm_data.loc[max_area, 'Dist_Name'])

# Phường nào có dân số 2019 (Pop_2019) cao nhất

max_pop= hcm_data['Pop_2019'].idxmax()
print("Phường có dân số 2019 (Pop_2019) cao nhất là phường:", hcm_data.loc[max_pop,'Com_Name'], "quận", hcm_data.loc[max_pop, 'Dist_Name'])

# Phường nào có diện tích nhỏ nhất
min_area= hcm_data['Shape_Area'].idxmin()
print("Phường có diện tích nhỏ nhất là phường:", hcm_data.loc[min_area,'Com_Name'], "quận", hcm_data.loc[min_area, 'Dist_Name'])

# Phường nào có dân số thấp nhất (2019)

min_pop= hcm_data['Pop_2019'].idxmin()
print("Phường có dân số 2019 (Pop_2019) thấp nhất là phường:", hcm_data.loc[min_pop,'Com_Name'], "quận", hcm_data.loc[min_pop, 'Dist_Name'])

# Phường nào có tốc độ tăng trưởng dân số nhanh nhất (dựa trên Pop_2009 và Pop_2019)

max_growth=(hcm_data['Pop_2019'] / hcm_data['Pop_2009']).idxmax()
print("Phường có tốc độ tăng trưởng dân số nhanh nhất là phường:", hcm_data.loc[max_growth,'Com_Name'], "quận", hcm_data.loc[max_growth, 'Dist_Name'])

# Phường nào có tốc độ tăng trưởng dân số thấp nhất

min_growth=(hcm_data['Pop_2019'] / hcm_data['Pop_2009']).idxmin()
print("Phường có tốc độ tăng trưởng dân số thấp nhất là phường:", hcm_data.loc[min_growth,'Com_Name'], "quận", hcm_data.loc[min_growth, 'Dist_Name'])

# Phường nào có biến động dân số nhanh nhất

max_volatility = (hcm_data['Pop_2019'] - hcm_data['Pop_2009']).idxmax()
print("Phường có biến động dân số nhanh nhất là phường:", hcm_data.loc[max_volatility,'Com_Name'], "quận", hcm_data.loc[max_volatility, 'Dist_Name'])
# Phường nào có biến động dân số chậm nhất
min_volatility = (hcm_data['Pop_2019'] - hcm_data['Pop_2009']).idxmin()
print("Phường có biến động dân số chậm nhất là phường:", hcm_data.loc[min_volatility,'Com_Name'], "quận", hcm_data.loc[min_volatility, 'Dist_Name'])

# Phường nào có mật độ dân số cao nhất (2019)

max_density = hcm_data['Den_2019'].idxmax()
print('Phường có mật độ dân số cao nhất (2019) là phường: ', hcm_data.loc[max_density, 'Com_Name'], "quận", hcm_data.loc[max_density, 'Dist_Name'])
# Phường nào có mật độ dân số thấp nhất (2019)

min_density = hcm_data['Den_2019'].idxmin()
print('Phường có mật độ dân số thấp nhất (2019) là phường: ', hcm_data.loc[min_density, 'Com_Name'], "quận", hcm_data.loc[min_density, 'Dist_Name'])