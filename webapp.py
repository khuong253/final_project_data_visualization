import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly as py
import ast
from PIL import Image

df = pd.read_csv('D:/HCMUS/ki2/trucquan/doan/data_science_asia_clean.csv',index_col=0)

def convert_to_tuple(value):
    if pd.notna(value):
        return ast.literal_eval(value)
    else:
        return value
for col in df.columns:
    check_value = df[col].iloc[0]
    if (pd.isnull(check_value)):
        i = 1
        while(pd.isnull(check_value)):
            check_value = df[col].iloc[i]
            i += 1
    if isinstance(check_value, str):
        if(check_value[0] == '('):
            df[col] = df[col].apply(convert_to_tuple)



st.title('Tình trạng về ngành nghề Data Science ở 4 nước Việt Nam, Nhật Bản, Hàn Quốc, Trung Quốc')

st.header('I. Giới thiệu chung')
image = Image.open('D:/HCMUS/ki2/trucquan/doan//datascience.jpg')
st.image(image)
st.write(
    '''
    Ngành Data Science và Machine Learning đang trở thành những lĩnh vực nổi bật và hấp dẫn trong thời đại công nghệ hiện đại. 
    Với sự gia tăng vượt bậc về dữ liệu và khả năng tính toán, việc áp dụng khoa học 
    dữ liệu và học máy đã trở thành công cụ mạnh mẽ để tìm ra những thông tin quan trọng và mô hình dự đoán chính xác.
    '''
)
st.write(
    '''
    Data Science là
      một ngành tập trung vào việc nghiên cứu và áp dụng các phương pháp, 
      công cụ và thuật toán để phân tích và hiểu sâu hơn về dữ liệu. Nó kết hợp các lĩnh vực như thống kê, khoa học máy, 
      và trí tuệ nhân tạo để xác định các mẫu, quy luật, và thông tin tiềm ẩn trong dữ liệu. Data Science được xem là một trong những nghề hot nhất 
      trong thế kỉ 21.
    '''
)

st.write(
    '''
    Ngành Data Science ở Châu Á đang trở thành một trong những ngành công nghệ phát triển nhanh nhất 
    và hứa hẹn mang lại nhiều cơ hội và tiềm năng lớn cho khu vực này. Với sự gia tăng về dữ liệu và 
    sự phổ biến của công nghệ thông tin, các quốc gia Châu Á đã nhận ra tầm quan trọng của việc áp dụng khoa học dữ liệu 
    và học máy để tận dụng tiềm năng của dữ liệu lớn.
    '''
)

st.write(
    '''
    Các quốc gia như Trung Quốc, Ấn Độ, Nhật Bản và Hàn Quốc đang dẫn đầu trong lĩnh vực Data Science ở Châu Á. 
    Trung Quốc, với dân số đông đúc và dữ liệu khổng lồ, đã đầu tư mạnh mẽ vào phát triển ngành này và
      trở thành một trong những trung tâm nghiên cứu và ứng dụng hàng đầu thế giới. Nhật Bản và Hàn Quốc 
      có một nền tảng công nghệ mạnh mẽ và các công ty công nghệ hàng đầu, và đang tìm cách tận dụng sức mạnh
      của Data Science để nâng cao hiệu suất kinh doanh và tạo ra giá trị mới.
    '''
)

st.write(
    '''
    Ở Việt Nam, ngành Data Science đang ngày càng thu hút sự quan tâm và phát triển nhanh chóng. Nhiều trường đại học và viện 
    nghiên cứu đã mở các chương trình đào tạo về Data Science và Machine Learning, nhằm cung cấp kiến thức và kỹ năng cần thiết
    trong lĩnh vực này. Tuy trong thị trường nghề nghiệp vẫn chưa có nhiều cơ hội nhưng với sự  quan tâm và đầu tư vào lĩnh vực 
    này đang gia tăng, và sự hỗ trợ từ cộng đồng, doanh nghiệp và chính phủ hứa hẹn mang lại nhiều cơ hội việc làm và tiềm năng tăng trưởng.

    Ta sẽ xem thử rằng tình trạng của ngành Data Science ở 4 nước Đông Á này ra sao từ đó đưa ra các góc nhìn và phân tích để hiểu rõ hơn về thị trường
    và tiềm năng phát triển của 4 nước này
    '''
)


st.header('II. Phân tích về ngành Data Science ở 4 quốc gia: Việt Nam, Nhật Bản, Hàn Quốc, Trung Quốc')

st.subheader('1. Phân tích tổng quát về ngành khoa học dữ liệu của 4 quốc gia')

st.markdown(

    '#### 1.1 Phân tích các ngôn ngữ lập trình phổ biển dùng cho khoa học dữ liệu'
)
#Tỷ lệ phần trăm các ngôn ngữ lập trình dùng cho khoa học dữ liệu theo khảo sát'

### Code 1.1
def create_new_array(fst, scd): 
    return np.append(fst, scd)
df_sub = df.copy()
new_arr = [(np.array(tuple)) for tuple in df_sub[df_sub.columns[9]]]
language_arr = new_arr[0]
for i in range(1, len(new_arr)):
    language_arr = create_new_array(language_arr, new_arr[i])
unique_languages =  np.unique(language_arr, return_counts=False)

# xóa nan
unique_languages = np.delete(unique_languages, np.where(unique_languages == "nan"))
count_list = []
check_for_nan = df_sub[df_sub.columns[9]].isnull()
for language in unique_languages:
    count = 0
    for tuple, is_nan in zip(df_sub[df_sub.columns[9]], check_for_nan):
        if is_nan == False and language in tuple:
            count += 1
    count_list.append(count)

sum_counts = sum(count_list)
language_rate_list = [round((val/sum_counts)*100, 2) for val in count_list]

language_rate_matrix = np.matrix([unique_languages, language_rate_list])
language_rate_matrix = language_rate_matrix.transpose()


language_rate_df = pd.DataFrame(language_rate_matrix, columns=['Programming language', 'Percentage'])
language_rate_df['Percentage'] = language_rate_df['Percentage'].astype(float)
language_rate_df = language_rate_df.sort_values(by=['Percentage'], ascending=True)

fig, ax = plt.subplots(figsize =(16, 9))
language = language_rate_df['Programming language']
percentage = language_rate_df['Percentage']
ax.barh(language, percentage)

ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)

#  Add x, y gridlines
ax.grid(visible= True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width(), i.get_y() + 0.1,
             str(i.get_width()),
             fontsize = 10, fontweight ='bold',
             color ='grey')

# Add Plot Title
ax.set_title('Tỷ lệ phần trăm các ngôn ngữ lập trình dùng cho khoa học dữ liệu theo khảo sát',
             loc ='left', )
ax.set_ylabel("Ngôn ngữ lập trình")
ax.set_xlabel("Tỷ lệ (%)")
st.pyplot(fig)


###

#Nhận xét 1.1
st.write('''Nhận xét:
- Nhìn vào biểu đồ, có thể thấy Python là ngôn ngữ lập trình được sử dụng nhiều nhất trong lĩnh vực khoa học dữ liệu ở 4 quốc gia khảo sát với tỷ lệ 33,92%. Đứng thứ hai là SQL với 11,1%, thứ ba là C++ với 10.32%. Các ngôn ngữ lập trình có tỷ lệ sử dụng ít nhất là Julia (0.67%), Go (1.18%), PHP (2.36%).

- Có thể nói Python chiếm tỷ lệ áp đảo là nhờ vào các thư viện được thiết kế cho khoa học dữ liệu như Numpy, Tensorflow, Keras, Scikit-Learn, v.v, giúp mọi giai đoạn phân tích dữ liệu trở nên thuận tiện và dễ dàng hơn. Ngoài ra cú pháp đơn giản, dễ đọc, dễ viết và có tốc độ thực thi nhanh hơn nhiều so với các ngôn ngữ lập trình khác. Python cũng là ngôn ngữ được ưu tiên sử dụng dành cho người mới bắt đầu học khoa học dữ liệu

- SQL là ngôn ngữ cần thiết cho quá trình truy cập cơ sở dữ liệu và thực hiện các thao tác với dữ liệu. SQL giữ vai trò quan trong trong khoa học dữ liệu.

- C++ là ngôn ngữ lập trình đa năng, có thể hỗ trợ làm việc với khối lượng dữ liệu lớn một cách hiệu quả và nhanh chóng.''')

### 1.2
st.markdown(

    '#### 1.2 Phân tích các IDE phổ biển dùng cho khoa học dữ liệu'
)
#Biểu đồ Môi trường phát triển tích hợp (IDE\'s) được sử dụng thường xuyên
###Code 1.2
# Lấy dữ liệu cột và tính số lượng từng loại IDE

df_IDE = df_sub['Which of the following integrated development environments (IDE\'s) do you use on a regular basis?'].explode().value_counts().to_dict()
# Loại bỏ các hàng dữ liệu có giá trị None
del df_IDE['None']

ides = ['Jupyter Notebook',
        'Visual Studio Code (VSCode)',
        'PyCharm','Visual Studio',
        'JupyterLab', 'MATLAB','RStudio',
        'Notepad++','Vim/Emacs','Spyder',
        'IntelliJ','Sublime Text', 'Other']

ides_value = []
for i in df_IDE:
    ides_value.append(df_IDE[i])

# Tạo biểu đồ cột ngang
fig = plt.figure(figsize=(16, 8))
ides_plot = plt.barh(ides, ides_value, color="#79CDCD")

# Đảo ngược trục y để các nền tảng có số người chọn nhiều nhất nằm ở phía trên
ax = plt.gca()
ax.invert_yaxis()
ax.bar_label(ides_plot, padding=4)

# Đặt tên cho trục x, trục y và tên biểu đồ
ax.set_ylabel("IDE\'s")
ax.set_xlabel("Số lượng")
plt.title("Biểu đồ Môi trường phát triển tích hợp (IDE\'s) được sử dụng thường xuyên", loc="center", fontsize=15)
st.pyplot(fig)
### Nhận xét 1.2
st.write('''Nhận xét:
- Trong biểu đồ trên, Jupyter Notebook và Visual Studio Code (VSCode) là hai môi trường phát triển phổ biến nhất, với số lượng người sử dụng lần lượt là 830 và 717. Điều này cho thấy sự ưa chuộng và độ phổ biến của hai IDE này trong việc phân tích dữ liệu và lập trình. Jupyter Notebook được ưa chuộng vì tính linh hoạt, khả năng tương tác và khả năng chia sẻ mã nguồn một cách dễ dàng, còn VSCode cho thấy tính linh hoạt và khả năng mở rộng của mình đã thu hút sự quan tâm của nhiều lập trình viên.
    - PyCharm, Visual Studio và JupyterLab cũng được sử dụng khá phổ biến, với số lượng người sử dụng lần lượt là 450, 326 và 302. Điều này cũng phần nào phản ánh chất lượng các tính năng, công cụ mà các IDE này hỗ trợ cho lập trình viên.
    - MATLAB, RStudio và Notepad++ cũng có mức độ sử dụng tương đối cao, với số lượng người sử dụng từ 142 đến 177.
    - Các môi trường phát triển như Vim / Emacs, Spyder, IntelliJ và Sublime Text có mức độ sử dụng thấp hơn so với các IDE trên.
    - Các môi trường phát triển khác: Có một số người dùng sử dụng các môi trường phát triển không thuộc danh sách trên, được ghi chung vào nhãn "Other" với số lượng là 72. Điều này cho thấy sự đa dạng trong lựa chọn IDE của người dùng.  
    
Tổng quan, các lựa chọn IDE của người dùng phụ thuộc vào nhiều yếu tố như tính linh hoạt, khả năng mở rộng, tính năng chuyên dụng và yêu cầu công việc cụ thể. Sự đa dạng trong lựa chọn IDE phản ánh sự đa dạng trong phong cách làm việc và ưu tiên của các lập trình viên và nhà phân tích dữ liệu.''')

###1.3
st.markdown(

    '#### 1.3 Phân tích các thư viện phổ biển dùng cho khoa học dữ liệu'
)
#Biểu đồ Thư viện trực quan hóa dữ liệu được sử dụng thường xuyên

### Code 1.3
# Lấy dữ liệu cột và tính số lượng từng thư viện
df_lib = df_sub['Do you use any of the following data visualization libraries on a regular basis?'].explode().value_counts().to_dict()

# Loại bỏ các hàng dữ liệu có giá trị None
del df_lib['None']

libs = ['Matplotlib',
        'Seaborn',
        'Plotly/Plotly Express',
        'Ggplot/ggplot2',
        'Geoplotlib',
        'D3 js',
        'Bokeh',
        'Other',
        'Shiny',
        'Leaflet/Folium',
        'Pygal',
        'Altair',
        'Highcharter',
        'Dygraphs']

libs_value = []
for i in df_lib:
    libs_value.append(df_lib[i])

# Tạo biểu đồ cột ngang
fig = plt.figure(figsize=(16, 8))
libs_plot = plt.barh(libs, libs_value, color="#8B8970")

# Đảo ngược trục y để các nền tảng có số người chọn nhiều nhất nằm ở phía trên
ax = plt.gca()
ax.invert_yaxis()
ax.bar_label(libs_plot, padding=4)

# Đặt tên cho trục x, trục y và tên biểu đồ

ax.set_ylabel("Thư viện trực quan hóa dữ liệu")
ax.set_xlabel("Số lượng")
plt.title("Biểu đồ Thư viện trực quan hóa dữ liệu được sử dụng thường xuyên", loc="center", fontsize=15)
st.pyplot(fig)

### 1.3 Nhận xét
st.write('''Nhận xét:
- Matplotlib là thư viện phổ biến nhất với 923 lượt đề cập. Matplotlib là một thư viện trực quan hóa dữ liệu đa nền tảng và mạnh mẽ trong Python.

- Seaborn là thư viện trực quan hóa dữ liệu thứ hai phổ biến nhất với 611 lượt đề cập. Seaborn cung cấp các chức năng cao cấp cho việc trực quan hóa dữ liệu thống kê và có tích hợp tốt với Pandas.

- Plotly/Plotly Express xếp thứ ba với 218 lượt đề cập. Plotly cung cấp các công cụ mạnh mẽ cho việc tạo ra các đồ thị tương tác và đồ họa web.

- Ggplot/ggplot2, với 200 lượt đề cập, là một thư viện trực quan hóa dữ liệu phổ biến trong R. Nó dựa trên cú pháp "Grammar of Graphics" để tạo ra các đồ thị chất lượng cao.

- Các thư viện như Geoplotlib, D3.js, Bokeh, Shiny và Leaflet/Folium cũng được đề cập trong số lần nhưng có số lượng thấp hơn so với các thư viện khác. Chúng được sử dụng chủ yếu cho việc trực quan hóa dữ liệu địa lý và dữ liệu tương tác.

- Các thư viện Pygal, Altair, Highcharter và Dygraphs cũng được đề cập nhưng có số lượng thấp. Chúng cung cấp các công cụ trực quan hóa dữ liệu đa dạng nhưng ít phổ biến hơn so với các thư viện khác.

Tổng quan, Matplotlib và Seaborn là hai thư viện phổ biến nhất cho trực quan hóa dữ liệu trong Python, trong khi Plotly cung cấp tính năng tương tác cao. Ggplot/ggplot2 là thư viện phổ biến trong R. Sự lựa chọn thư viện trực quan hóa dữ liệu cụ thể phụ thuộc vào yêu cầu của bạn và sự thoải mái với các công cụ cụ thể.''')

### 1.4
st.markdown(

    '#### 1.4 Phân tích xem những framework Machine Learning phổ biến thường được sử dụng với những thuật toán Machine Learning nào.'
)
#Technique Usage by Libraries
# 1.4 code

split_df = df_sub[['Which of the following machine learning frameworks do you use on a regular basis?','Which of the following ML algorithms do you use on a regular basis?']].dropna()
dict_ml_frameworks = split_df['Which of the following machine learning frameworks do you use on a regular basis?'].explode().value_counts().to_dict()
top4_ml_frameworks = list(dict_ml_frameworks.keys())[0:4]

dict_ml_frameworks_algorithms= {}
for i in top4_ml_frameworks:
    flattened_values = split_df[split_df['Which of the following machine learning frameworks do you use on a regular basis?'].apply(lambda x: any(i in item for item in x))]['Which of the following ML algorithms do you use on a regular basis?'].explode()
    dict_ml_frameworks_algorithms[i] = flattened_values.value_counts().to_dict()

# Mapping of full names to abbreviations
name_to_abbrev = {
    'Convolutional Neural Networks': 'CNN',
    'Linear or Logistic Regression': 'LLR',
    'Decision Trees or Random Forests': 'DT/RF',
    'Gradient Boosting Machines (xgboost, lightgbm, etc)': 'GBM',
    'Dense Neural Networks (MLPs, etc)': 'DNN',
    'Transformer Networks (BERT, gpt-3, etc)': 'TN',
    'Recurrent Neural Networks': 'RNN',
    'Bayesian Approaches': 'BA',
    'Autoencoder Networks (DAE, VAE, etc)': 'AE',
    'Generative Adversarial Networks': 'GAN',
    'Graph Neural Networks': 'GNN',
    'Evolutionary Approaches': 'EA',
    'None': 'None',
    'Other': 'Other'
}

# Delete None values
for library in dict_ml_frameworks_algorithms:
    del dict_ml_frameworks_algorithms[library]['None']

# Extract the techniques and libraries
techniques = list(dict_ml_frameworks_algorithms[list(dict_ml_frameworks_algorithms.keys())[0]].keys())
libraries = list(dict_ml_frameworks_algorithms.keys())

# Set the position of each technique on the x-axis
x = np.arange(len(techniques))

# Set the width and size of the bars
width = 0.2
fig = plt.figure(figsize=(14,8))

# Plotting the bars for each library
plt.bar(x - width, [dict_ml_frameworks_algorithms[libraries[0]][technique] for technique in techniques], width, label=libraries[0])
plt.bar(x, [dict_ml_frameworks_algorithms[libraries[1]][technique] for technique in techniques], width, label=libraries[1])
plt.bar(x + width, [dict_ml_frameworks_algorithms[libraries[2]][technique] for technique in techniques], width, label=libraries[2])
plt.bar(x + 2 * width, [dict_ml_frameworks_algorithms[libraries[3]][technique] for technique in techniques], width, label=libraries[3])

# Customize the x-axis tick labels with abbreviations
abbreviations = [name_to_abbrev[technique] for technique in techniques]
plt.xticks(x, abbreviations, rotation=90)

# Set the title and labels
plt.title('Technique Usage by Libraries')
plt.xlabel('Techniques')
plt.ylabel('Count')
plt.ylim(0, 500)

# Create a legend for the libraries
plt.legend()

plt.tight_layout()

st.pyplot(fig)


st.write('''Nhận xét:
- Scikit-learn và PyTorch là hai thư viện phổ biến được sử dụng nhiều nhất trong cộng đồng Machine Learning và Deep Learning.

- TensorFlow và Keras cũng được sử dụng khá phổ biến, tuy nhiên không đạt được mức độ sử dụng như Scikit-learn và PyTorch.

Tổng quan, Scikit-learn và PyTorch là hai thư viện phổ biến được sử dụng rộng rãi với một loạt các thuật toán học máy khác nhau. TensorFlow và Keras, trong khi không sử dụng nhiều thuật toán truyền thống như DT/RF và GBM, lại tập trung nhiều vào Deep Learning với CNN, DNN và RNN. Trong khi đó, Scikit-learn có sự đa dạng hơn với sự sử dụng của các thuật toán truyền thống và cũng hỗ trợ một số thuật toán Deep Learning.''')

#1.5
st.markdown(

    '#### 1.5 Phân tích thuật toán Machine Learning phổ biển dùng cho khoa học dữ liệu theo năm kinh nghiệm'
)
#Tỷ lệ sử dụng các thuật toán Machine Learning

#Code 1.5

new_arr = [(np.array(tuple)) for tuple in df_sub[df_sub.columns[15]]]
ML_algo_arr = new_arr[0]
for i in range(1, len(new_arr)):
    ML_algo_arr = create_new_array(ML_algo_arr, new_arr[i])
unique_ML_algos =  np.unique(ML_algo_arr, return_counts=False)

# xóa nan, none
unique_ML_algos = np.delete(unique_ML_algos, np.where(unique_ML_algos == "None"))
unique_ML_algos = np.delete(unique_ML_algos, np.where(unique_ML_algos == "nan"))


unique_ML_exp = ['Under 1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years',
       '5-10 years', '10-20 years']

matrix = []
for ML_algo in unique_ML_algos:
    row = []
    for year_exp in unique_ML_exp:
        tmp_df = df.query("`For how many years have you used machine learning methods?` == @year_exp")
        check_for_nan = tmp_df[tmp_df.columns[15]].isnull()

        total_row = tmp_df[tmp_df.columns[15]].count()
        count = 0
        for data, is_nan in zip(tmp_df[tmp_df.columns[15]], check_for_nan):
            if is_nan == True:
                continue
            if ML_algo in data:
                count += 1
        row.append(count/total_row)
    matrix.append(row)

df_visual = pd.DataFrame(matrix,unique_ML_algos , unique_ML_exp)

fig = plt.figure(figsize=(16,8))
sns.heatmap(df_visual, cmap = "Greens")
plt.title("Tỷ lệ sử dụng các thuật toán Machine Learning")
plt.xlabel("Số năm sử dụng Machine Learning")
plt.ylabel("Thuật toán Machine Learning")
plt.rc('font', size=16)

st.pyplot(fig)
st.write('''Nhận xét:
         
- Nhìn vào biểu đồ, có thể thấy những nhà khoa học dữ liệu càng có nhiều năm kinh nghiệm sử dụng Machine Learning thì tương ứng với đó là khả năng sử dụng/thành thạo nhiều thuật toán Machine Learning hơn những người mới bắt đầu.

- Linear or Logistic Regression, Convolutional Neural Networks, Decision Trees or Random Forests, Gradient Boosting Machines (xgboost, lightgbm, etc) là những thuật toán thường được sử dụng nhiều nhất từ dưới 1 năm kinh nghiệm đến 10-20 năm kinh nghiệm, trong đó Linear or Logistic Regression có tỷ lệ sử dụng cao nhất. Những thuật toán này là những thuật toán cơ bản trong Machine Learning nên có thể dễ dàng sử dụng và mang lại hiệu quả cao ngay từ những năm đầu kinh nghiệm.

- Các thuật toán Autoencoder Networks (DAE, VAE, etc), Evolutionary Approaches, Generative Adversarial Networks, Graph Neural Networks là những thuật toán ít được sử dụng, đặc biệt đối với những người dưới 4 năm kinh nghiệm, tuy nhiên, với những người hơn 4 năm kinh nghiệm, các thuật toán này được sử dụng nhiều hơn. Có thể do các thuật toán này có độ phức tạp cao, đòi hỏi nhiều thời gian nghiên cứu mới có thể sử dụng hiệu quả.
         '''
         )

## 1.6
st.markdown(

    '#### 1.6 Phân tích tương quan số năm kinh nghiệm và công cụ IDE thường dùng'
)

### 1.6 Code
# Lọc các hàng có ô IDE giá trị None
df_new = df.dropna(subset=['Which of the following integrated development environments (IDE\'s) do you use on a regular basis?'])
df_new = df_new[df_new["Which of the following integrated development environments (IDE\'s) do you use on a regular basis?"].apply(lambda x: x != ('None',))]

# Lấy ra dữ liệu 2 cột cần thiết
df_exp = df_new[['For how many years have you been writing code and/or programming?', 'Which of the following integrated development environments (IDE\'s) do you use on a regular basis?']]

# Sắp xếp lại theo mốc kinh nghiệm
exp_count = df_exp.explode('Which of the following integrated development environments (IDE\'s) do you use on a regular basis?').value_counts().reset_index(name='Freq')
values_time = ['< 1 years','1-3 years', '3-5 years', '5-10 years', '10-20 years','20+ years']
exp_count['For how many years have you been writing code and/or programming?'] = pd.Categorical(exp_count['For how many years have you been writing code and/or programming?'], categories=values_time, ordered=True)
exp_count = exp_count.sort_values(by='For how many years have you been writing code and/or programming?')

#Vẽ biểu đồ
exp_pivot = exp_count.pivot(index='Which of the following integrated development environments (IDE\'s) do you use on a regular basis?', 
columns='For how many years have you been writing code and/or programming?', 
values=  'Freq')


fig,ax = plt.subplots(figsize=(16,8))
ax = sns.heatmap(exp_pivot, cmap='Blues')
ax.set(xlabel="Kinh nghiệm thực hành", ylabel= "IDE's")
plt.title("Biểu đồ sử dụng Môi trường phát triển tích hợp theo số năm kinh nghiệm", fontsize=15)

st.pyplot(fig)


#Biểu đồ sử dụng Môi trường phát triển tích hợp theo số năm kinh nghiệm
st.write('''Nhận xét:
- Biểu đồ trên thể hiện sự phân bố việc sử dụng công cụ IDE theo kinh nghiệm lập trình của người được khảo sát.
- Ta có thể thấy Jupyter Notebook và VSCode là 2 dải màu nổi nhất trong biểu đồ, điều này cho thấy tính ưa chuộng của phần lớn lập trình viên 4 nước châu Á (Trung Quốc, Việt Nam, Hàn Quốc, Nhật Bản) đối với 2 ứng dụng này và cũng cho thấy sự bền bỉ, chất lượng khi có sự phổ biến mạnh mẽ trong từng nhóm kinh nghiệm.
- Các ứng dụng IDE có dải màu nổi bật tiếp theo là Visual Studio và Pycharm, tuy kém phổ biến hơn 2 ứng dụng trên nhưng cũng là lựa chọn hàng đầu của các lập trình viên, đặc biệt trong nhóm kinh nghiệm 10 năm trở xuống.
- Các IDE còn lại đều có người sử dụng nhưng sẽ được dùng trong lĩnh vực, tình huống nhất định nên khả năng hấp dẫn kém hơn so với các IDE trên.

Tóm lại, tùy theo kinh nghiệm cá nhân và tình huống sử dụng mà lập trình viên, người dùng sẽ chọn IDE phù hợp cho công việc của mình. Nhưng các IDE sau vẫn sẽ là sự lựa chọn hàng đầu của các lập trình viên: Jupyter Notebook, VSCode.''')

## 1.7
st.markdown(

    '#### 1.7 Phân tích xem những người có trình độ học vấn cao hơn có xu hướng sử dụng những product hoặc nền tảng nào khi bắt đầu học data science'
)

### 1.7 code

df_1 = df[['What products or platforms did you find to be most helpful when you first started studying data science?',"What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"]].dropna()

arr_level = df_1["What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"].unique()

dict_level_products_platforms= {}
for i in arr_level:
    flattened_values = df_1[df_1["What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"]==i]['What products or platforms did you find to be most helpful when you first started studying data science?'].explode()
    dict_level_products_platforms[i] = flattened_values.value_counts().to_dict()

# Delete None values
for library in dict_level_products_platforms:
    del dict_level_products_platforms[library]['None / I do not study data science']
del dict_level_products_platforms['I prefer not to answer']

level = ['Doctoral degree',
'Professional doctorate',
'Master’s degree',
'Bachelor’s degree',
'Some college/university study without earning a bachelor’s degree',
'No formal education past high school',
'I prefer not to answer',]

education_levels = ['Doctoral degree',
'Professional doctorate',
'Master’s degree',
'Bachelor’s degree',
'Some college/university study without earning a bachelor’s degree',
'No formal education past high school']
product_frequencies = []
for education_level in education_levels:
    product_frequencies.append(list(dict_level_products_platforms[education_level].values()))

# Vẽ biểu đồ đường
plt.rc('font', size=12)
fig = plt.figure(figsize=(16, 8))

for i in range(len(product_frequencies[0])):
    frequencies = [product[i] for product in product_frequencies]
    plt.plot(education_levels, frequencies, marker='o', label=list(list(dict_level_products_platforms.values())[0].keys())[i])

plt.xlabel('Education Level')
plt.ylabel('Frequency')
plt.title('Frequency of Useful Products Used When Starting Data Science by Education Level')
plt.legend()
short_levels = ['Doctoral degree',
'Professional doctorate',
'Master’s degree',
'Bachelor’s degree',
'College/university without a bachelor',
'Past high school']
plt.xticks(education_levels, short_levels, rotation=90)
plt.xticks( rotation=45)
st.pyplot(fig)
#Frequency of Useful Products Used When Starting Data Science by Education Level
st.write('''Nhận xét:
- Từ biểu đồ này, ta thấy được sự chênh lệch giữa trình độ học vấn khi tham gia cuộc khảo sát. Những người tham gia cuộc khảo sát này đa số đều đã trải qua quá trình học đại học. Có vẻ những người học nghành Data Science đều cần học vấn khá cao để làm việc (đa số đều từ bậc Bachelor’s degree trở lên) nên rất ích người chưa hoàn thành chương trình đại học.

- Các sản phẩm phổ biến nhất cho tất cả các mức trình độ học vấn là "Kaggle (notebooks, competitions, etc)", "Online courses (Coursera, EdX, etc)", và "Video platforms (YouTube, Twitch, etc)". Đây là những nguồn học tập vô cùng phổ biến ở trong giới Data Science bới sự tiện dụng và đa dạng trong tài liệu và chủ đề.

- Thứ tự mức độ sử dụng của các nền tảng ở mọi nền tảng của tất cả trình độ học vấn đều như nhau, chứng tỏ không có nền tảng nào ít được sử dụng ở trình độ này nhưng được sử dụng nhiều ở trình độ khác.
''')

## 2
st.subheader(

    'II So sánh về ngành khoa học dữ liệu của 4 quốc gia.'
)




st.markdown(

    '#### 2.1 Phân tích những người học Data Science theo độ tuổi của 4 quốc gia'
)

## 2.1 Code
age = 'What is your age (# years)?'
gender = 'What is your gender'
country = 'In which country do you currently reside?'
education = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'


def age_distribute(ax, country,data):
    data_age = df[age].value_counts().sort_index()
    ax.bar(data.index, data, width=0.55, 
       edgecolor='darkgray', color='#d4dddd',
       linewidth=0.7)
    ax.set_title('Age Distribution of {}'.format(country))
    ax.grid(axis='y', linestyle='-', alpha=0.5)

age_country = df.groupby([country,age])[age].count()

fig, ax = plt.subplots(2,2, figsize=(15, 10))
ax_flat = ax.flatten()
age_distribute(ax_flat[0],'China',age_country.loc['China'])
age_distribute(ax_flat[1],'Viet Nam',age_country.loc['Viet Nam'])
age_distribute(ax_flat[2],'South Korea',age_country.loc['South Korea'])
age_distribute(ax_flat[3],'Japan',age_country.loc['Japan'])

st.pyplot(fig)
#Age Distribution of Chine, VietNam, South Korea, Japan/ Bar
st.write('''Nhận xét:
- Ta nhận thấy ở Việt Nam và Trung Quốc thì số lượng lao động trẻ rất cao (từ 18 - 24), trong khi đó số lượng lao động có kinh nghiệm với giảm nhanh và ít hơn rất nhiều. Điều này chứng tỏ đây là 2 thị trường khá là mới và tiềm năng của ngành này. Tuy nhiên ở thị trường Việt Nam thì số lượng lao động có kinh nghiệm lại ít hơn khá nhiều nên lo sợ việc chất lượng lao động được đào tạo bài bản không cao.
- Ở Hàn và Nhật thì lại trái ngược lại khi số lượng lao động có kinh nghiệm lại nhiều hơn số lượng lao động trẻ, có thể là do xu hướng già hoá dân số ở hai nước này và có vẻ như ngành DS đã qua thời kì hot ở hai nước này. Tuy nhiên số lượng lao động có kinh nghiệm của họ lại rất nhiều, cho thấy rằng chất lượng nhân sự của họ rất tốt, đây sẽ là hai thị trường tuy không còn hot nhưng chất lượng lại vô cùng tốt.
''')
#Age Distribution of Chine, VietNam, South Korea, Japan/ Pie
edu = df.groupby([country,education])[education].count()

def pie_pro(ax, country,data):
    sns.set_style("whitegrid")  # Đặt phong cách
    custom_colors = sns.color_palette("Set3", len(data))
    ax.pie(data.values, colors=custom_colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    ax.set_title('Age Distribution of {}'.format(country))

fig, ax = plt.subplots(2,2, figsize=(10, 10))
ax_flat = ax.flatten()
pie_pro(ax_flat[0],'China',edu.loc['China'])
pie_pro(ax_flat[1],'Viet Nam',edu.loc['Viet Nam'])
pie_pro(ax_flat[2],'South Korea',edu.loc['South Korea'])
pie_pro(ax_flat[3],'Japan',edu.loc['Japan'])
plt.legend(edu.loc['China'].index,bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)

st.write('''Nhận xét:
- Ta nhận thấy rằng dù ở nước nào thì bằng Master đều chiếm phần lớn nhất, chứng tỏ rằng bằng Master là vô cùng quan trọng ở ngành DS. Tuy nhiên bằng tiến sĩ hay giáo sư lại ít hơn nhiều. Chứng tỏ bằng Master là để cho yêu cầu làm việc chứ không nhiều người muốn đi theo hướng nghiên cứu. Nhưng điều này cũng khẳng định rằng nếu bạn muốn làm trong ngành này thì yêu cầu về kiến thức là vô cùng cao.
- Ở Nhật Bản thì tỷ lệ người không học đại học nhưng ra làm ngành này là nhiều hơn hẳn so với 3 nước còn lại. Có thể ở thị trường Nhật Bản thì yêu cầu bằng cấp là không nhiều. Tuy nhiên tỷ lệ bằng Master cao chứng tỏ rằng nếu muốn có một công việc chắc ăn thì bạn cần học ở các cơ sở đào tạo chính quy''')

##2.2
st.markdown(

    '#### 2.2 Phân tích số lượt bình chọn cho nền tảng/sản phẩm hữu ích cho ngành khoa học dữ liệu giữa 4 quốc gia'
)
#Khảo sát bình chọn cho nền tảng/sản phẩm hữu ích để học khoa học dữ liệu ở bốn quốc gia
###2.2 code

def create_new_array(fst, scd): 
    return np.append(fst, scd)

new_arr = [(np.array(tuple)) for tuple in df[df.columns[5]]]
platform_arr = new_arr[0]
for i in range(1, len(new_arr)):
    platform_arr = create_new_array(platform_arr, new_arr[i])
unique_platforms =  np.unique(platform_arr, return_counts=False)

vn_df = df[(df[df.columns[3]] == 'Viet Nam')]
jp_df = df[(df[df.columns[3]] == 'Japan')]
cn_df = df[(df[df.columns[3]] == 'China')]
kr_df = df[(df[df.columns[3]] == 'South Korea')]
countries = []

def visualize(df):
    count_list = []
    
    for platform in unique_platforms:
        count = 0
        for tuple in df[df.columns[5]]:
            if platform in tuple:
                count += 1
        count_list.append(count)
    return count_list

vn_count = visualize(vn_df)
cn_count = visualize(cn_df)
jp_count = visualize(jp_df)
kr_count = visualize(kr_df)

unique_platforms[0] = unique_platforms[0][:unique_platforms[0].find(' (')]
unique_platforms[10] = unique_platforms[10][:unique_platforms[10].find(' (')]

# tạo dataframe
count_matrix = np.array([vn_count, cn_count, jp_count, kr_count])
vision_df = pd.DataFrame(count_matrix, columns=unique_platforms)
vision_df.insert(0, "Country", ['Viet Nam', 'China', 'Japan', 'Korea'], False)

# vẽ biểu đồ
vision_df.plot(x="Country", y=unique_platforms, kind="bar",figsize=(20,8))
plt.xticks(rotation = 'horizontal')
plt.title("Khảo sát bình chọn cho nền tảng/sản phẩm hữu ích cho ngành khoa học dữ liệu ở bốn quốc gia", fontsize=16)
plt.xlabel('Quốc gia', fontsize=16)
plt.ylabel('Số lượt bình chọn', fontsize=16)
plt.rc('font', size=12)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.write('''Nhận xét:
- Nhìn chung, đa số người tham gia khảo sát ở cả 4 quốc gia đều chọn các nền tảng nổi tiếng, uy tín như Coursera, Kaggle Learn Courses, University Courses để học khoa học dữ liệu. Bên cạnh đó, số lượng người chọn "Không" cũng chiếm số lượng khá lớn. Các nền tảng ít được chọn nhất là Fast.ai, Linkedin Learning, Udacity

- Tại Việt Nam, Coursera là nền tảng được nhiều lượt chọn nhất (hơn 100 lượt bình chọn), tiếp đến là học thông qua chương trình đại học (University Courses), sau đó là Udemy, DataCamp. Cloud-certification programs, Fast.ai, Udacity là những nền tảng ít được bình chọn nhất. Có thể thấy, người học khoa học dữ liệu ở Việt Nam vẫn ưu tiên lựa chọn các nền tảng nổi tiếng, uy tín, được nhiều người sử dụng trên thế giới hoặc là phương thức học truyền thống - chương trình đại học để trau dồi kỹ năng của mình.

- Tại Trung Quốc, University Courses là lựa chọn hàng đầu để học khoa học dữ liệu với hơn 180 lượt bình chọn, sau đó là Kaggle Learn Courses, Coursera, tuy nhiên, số lượt bình chọn cho "Không" cũng khá cao. Các nền tảng ít được sử dụng là Cloud-certification programs, Fast.ai, Udemy, Udacity, Linkedin Learning. Có thể thấy tại Trung Quốc, University Courses vẫn là lựa chọn ưu tiên hàng đầu để mọi người học khoa học dữ liệu.

- Tại Nhật bản, số lựa chọn "Không" chiếm số lượng lớn câu trả lời, tiếp đến là các nền tảng online như Kaggle Learn Courses, Udemy, Coursera. Lựa chọn thấp nhất thuộc về Linkedin Learning, Fast.ai, edX. Có thể thấy ở Nhật bản, nhiều người tham gia khảo sát chưa thấy các nền tảng/sản phẩm thật sự hữu ích. Mặt khác, nhiều người đánh giá Kaggle Learn Courses, Udemy, Coursera rất hữu ích và nổi trội hơn hẳn so với các nền tảng còn lại.

- Tại Hàn Quốc, University Courses là lựa chọn hàng đầu để học khoa học dữ liệu, sau đó là Coursera, Kaggle Learn Courses. Các nền tảng ít được sử dụng là Fast.ai, Linkedin Learning, Udacity, edX. Có thể thấy tại Hàn Quốc, University Courses vẫn là lựa chọn ưu tiên hàng đầu để mọi người học khoa học dữ liệu, tiếp đến là các nền tảng online như Coursera, Kaggle Learn Courses. Mặt khác, nhiều người tham gia khảo sát chưa thấy các nền tảng/sản phẩm thật sự hữu ích khi số lượt bình chọn "Không" vẫn khá cao.
''')

##3
st.subheader(

    'III Phân tích về ngành khoa học dữ liệu ở Việt Nam.'
)
#3.1
st.markdown(

    '#### 3.1 Khảo sát về nền tảng người tham gia khảo sát tại Việt Nam cảm thấy hữu ích nhất khi mới bắt đầu học Data Science'
)
#Nền tảng hữu ích khi mới bắt đầu học Data Science
#3.1 CODE

vn_df = df.loc[df['In which country do you currently reside?'] == 'Viet Nam' ]
vn_df.head(5)

count = vn_df.count()[1]

vn_df_platform = vn_df['What products or platforms did you find to be most helpful when you first started studying data science?'].explode().value_counts().to_dict()
del vn_df_platform['None / I do not study data science']

# x = platforms: Các nền tảng / ứng dụng được sử dụng khi mới bắt đầu học Data Science bởi người khảo sát tại Việt Nam
# y = platforms_value: Số lượng người sử dụng tương ứng

platforms = ['Kaggle', 'Online courses (Coursera, EdX, etc)', 
             'Video platforms (Youtube, Twitch, etc)',
             'University courses',
             'Social media platforms (Reddit, Twitter, etc)',
             'Other']

platforms_value = []
for i in vn_df_platform:
    platforms_value.append(vn_df_platform[i])

# Tạo biểu đồ cột ngang

fig = plt.figure(figsize=(16, 8))
platform_plot = plt.barh(platforms, platforms_value, color="lightgrey")

# Đảo ngược trục y để các nền tảng có số người chọn nhiều nhất nằm ở phía trên

ax = plt.gca()
ax.invert_yaxis()
ax.bar_label(platform_plot, padding=4)

# Đặt top 3 nền tảng có số người chọn nhiều nhất màu hồng để làm nổi bật

for i in range (0,3):
    ax.get_children()[i].set_color("pink")

# Đặt tên cho trục x, trục y và tên biểu đồ

ax.set_ylabel("Nền tảng")
ax.set_xlabel("Số lượng")
plt.title("Nền tảng hữu ích khi mới bắt đầu học Data Science", loc="left", fontsize=15)
st.pyplot(fig)



st.write('''Nhận xét:
- Kaggle là nền tảng được lựa chọn hữu ích nhất khi mới bắt đầu học Data Science bởi những người khảo sát đang sinh sống tại Việt Nam với 129 lượt chọn. Tiếp đến là các khoá học online như Coursera, EdX,... với 116 lượt chọn. Thứ ba là các nền tảng xem video trực tuyến như Youtube, Twitch,... với 107 lượt chọn.
- Có thể thấy các nền tảng học online nổi tiếng, có uy tín là sự lựa chọn hàng đầu đối với những người khảo sát tại Việt Nam (Kaggle, Coursera, EdX). Những nền tảng này có những bài giảng về Data Science bài bản, chuyên nghiệp, đầu tư, có thể chia sẻ kiến thức với nhiều người ở khắp mọi nơi trên thế giới, cung cấp chứng chỉ online đảm bảo kiến thức, là một trong những yếu tố được các công ty xem xét. Kaggle còn là nơi lưu trữ rất nhiều dữ liệu, được kiểm tra và đảm bảo tin cậy, có thể sử dụng để thực hành các dự án về Data Science.
- Nằm trong thời đại internet phát triển rộng rãi, không quá bất ngờ khi những nền tảng online là những lựa chọn nhiều nhất khi mới bắt đầu học Data Science. Các nền tảng online mang lại sự thuận tiện cho người học, có thể học bất cứ lúc nào, bất cứ đâu, có thể xem lại bài giảng, trao đổi online,... Các khoá học tại trường đại học chỉ xếp thứ 4 với 76 lượt chọn.
''')
#3.2
st.markdown(

    '#### 3.2 Khảo sát phân chia theo nhóm học sinh/sinh viên và nhóm không phải học sinh/sinh viên'
)
#Nền tảng hữu ích khi mới bắt đầu học Data Science
#code 3.2

vn_df_student = vn_df[['Are you currently a student?', 'What products or platforms did you find to be most helpful when you first started studying data science?']].dropna()
is_student = vn_df_student['Are you currently a student?'].unique()

student_count = vn_df_student.explode('What products or platforms did you find to be most helpful when you first started studying data science?').value_counts().reset_index(name='Freq')
student_count.drop(student_count[student_count['What products or platforms did you find to be most helpful when you first started studying data science?'] == 'None / I do not study data science'].index, inplace = True)
student_count.pivot_table('Freq', 'Are you currently a student?', 'What products or platforms did you find to be most helpful when you first started studying data science?').plot(kind='bar', figsize=(16,8))

# Đặt tên cho trục x, trục y và tên biểu đồ
ax = plt.subplot()
ax.set_ylabel("Số lượng")
ax.set_xlabel("Có đang là sinh viên không?")

plt.xticks(rotation=0)
plt.title("Nền tảng hữu ích khi mới bắt đầu học Data Science", fontsize=15)
plt.rc('font', size=12)
plt.legend()
st.pyplot()

student_pivot = student_count.pivot('What products or platforms did you find to be most helpful when you first started studying data science?', 'Are you currently a student?', 'Freq')


plt.subplots(figsize=(16,8))
ax = sns.heatmap(student_pivot, square=True, vmin=5, vmax=85)
ax.set(xlabel="Có đang là học sinh/sinh viên không?", ylabel= "Nền tảng")
plt.title("Nền tảng hữu ích khi mới bắt đầu học Data Science", fontsize=15)
st.pyplot()
st.write('''Nhận xét:
- Không có sự khác biệt nhiều giữa nhóm người đang là học sinh/sinh viên với nhóm người không là học sinh/sinh viên tại Việt Nam khi 3 lựa chọn nền tảng hữu ích khi mới bắt đầu học Data Science là Kaggle, các khoá học online (Coursera, EdX,...) và nền tảng video trực tuyến (YouTube, Twitch,...)
- Các nhóm người khảo sát đều cho rằng học qua các nền tảng trực tuyến chuyên nghiệp, bài bản là một sự lựa chọn hàng đầu. Cũng là nền tảng trực tuyến nhưng các trang mạng xã hội lại nhận được ít sự lựa chọn hơn rất nhiều. 
''')
##3.3
st.markdown(

    '#### 3.3 Khảo sát nền tảng người khảo sát hoàn thành khoá học Data Science'
)
#Nền tảng khi mới bắt đầu được sử dụng để hoàn thành khoá học Data Science
##code 3.3
vn_df_study = vn_df['On which platforms have you begun or completed data science courses?'].dropna()
vn_df_study = vn_df_study.explode().value_counts().to_dict()
del vn_df_study['None']

study_list = list(vn_df_study.keys())
study_list = [x.strip(' ') for x in study_list]

study_count = []
for i in vn_df_study:
    study_count.append(vn_df_study[i])

study_count_per = [round((val/count)*100,2) for val in study_count]

fig = plt.figure(figsize=(16,8))

ide_plot = plt.barh(study_list, study_count_per, color="grey")

# Đảo ngược trục y để các nền tảng có số người chọn nhiều nhất nằm ở phía trên

ax = plt.gca()
ax.invert_yaxis()
ax.bar_label(ide_plot, padding=4)

# Đặt top 3 nền tảng có số người chọn nhiều nhất màu hồng để làm nổi bật

for i in range (0,len(study_list)):
    if study_count_per[i] > 30:
        ax.get_children()[i].set_color("pink")

# Đặt tên cho trục x, trục y và tên biểu đồ

ax.set_ylabel("Nền tảng")
ax.set_xlabel("%")
plt.title("Nền tảng khi mới bắt đầu được sử dụng để hoàn thành khoá học Data Science", loc="left", fontsize=15)
st.pyplot(fig)

st.write('''Nhận xét:
- Coursera là nền tảng được dùng khi mới bắt đầu để hoàn thành khoá học Data Science nhiều nhất trong khảo sát với 49.53% người khảo sát lựa chọn, tiếp đến là các khoá học tại trường đại học với 35.85%, và thứ 3 là Kaggle Learn Courses với 31.13%
- Nhiều người khảo sát ở đây đã bắt đầu Data Science thông qua Coursera, một nền tảng nổi tiếng. Mặt khác, số người được tiếp cận Data Science thông qua các khoá học tại trường Đại học, hoặc theo học các ngành, chuyên ngành tại trường đại học cũng rất cao. Đây là cách tiếp cận truyền thống, bài bản và chuyên nghiệp.
- Kaggle Learn Courses, Udemy, LinkedIn Learning mặc dù ít được lựa chọn hơn Coursera nhưng đây cũng là những lựa chọn được nhiều người sử dụng để bắt đầu và hoàn thành khoá Data Science. Các chương trình chứng chỉ cloud, Fast.ai, Udacity là những lựa chọn chưa được phổ biến trong số người khảo sát tại Việt Nam.
''')


