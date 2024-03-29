import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import warnings
warnings.simplefilter("ignore")

df = pd.read_csv('data.csv', encoding="ISO-8859-1")
df_now=pd.read_csv('rfm_data.csv')
df_now['CustomerID'] = df_now['CustomerID'].astype(str)
df_ex=pd.read_csv('rfm_ex.csv')

def rfm_level(df):
  if df['R']==4 and df['F']==4 and df['M']==4:
    return 'Khach hang Vip'
  elif df['M']>=3 and df['F']>=2 and df['R']<=3:
    return 'Khach hang than thiet'
  elif df['R']==1 and df['F']==1 and df['M']==1:
    return 'Khach hang khong tham gia nua'
  else:
    return 'Regulars'



def get_customer_rfm(df_now, customer_id):
  # Lấy thông tin khách hàng
  customer_info = df_now.loc[df_now['CustomerID'] == customer_id]

  if customer_info.empty:
    return None

  frequency = customer_info['Frequency'].values[0]
  monetary = customer_info['Monetary'].values[0]
  recency = customer_info['Recency'].values[0]
  level=customer_info['RFM_Level'].values[0]

  # Trả về kết quả
  return frequency, monetary, recency,level












#GUI
st.title("Data Science & Machine Learning Project")
st.header("Customer Segmentation", divider='rainbow')

menu = ["Business Understanding", "Data Understanding","Data preparation","Modeling & Evaluation","Predict","Predict cho bất kỳ data nào (đưa về dạng RFM hoặc chưa)"] 
choice = st.sidebar.selectbox('Menu', menu)

def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file, encoding='latin-1', sep='\s+', header=None, names=['Customer_id', 'day', 'Quantity', 'Sales'])
        df.to_csv("data.csv", index=False)
        df['Date'] = pd.DatetimeIndex(df.InvoiceDate).date
        st.session_state['df'] = df
        return df
    else:
        st.write("Please upload a data file to proceed.")
        return None




# Main Menu
if choice == 'Business Understanding':
    st.subheader("Business Objective")
    st.write("""
     Phân khúc/nhóm/cụm khách hàng (market segmentation
– còn được gọi là phân khúc thị trường) là quá trình
nhóm các khách hàng lại với nhau dựa trên các đặc
điểm chung. Nó phân chia và nhóm khách hàng thành
các nhóm nhỏ theo đặc điểm địa lý, nhân khẩu học, tâm
lý học, hành vi (geographic, demographic,
psychographic, behavioral) và các đặc điểm khác..

     Các nhà tiếp thị sử dụng kỹ thuật này để nhắm mục tiêu
khách hàng thông qua việc cá nhân hóa, khi họ muốn
tung ra các chiến dịch quảng cáo, truyền thông, thiết kế
một ưu đãi hoặc khuyến mãi mới, và cũng để bán hàng.

     Tại sao cần “Customer Segmentation:
▪ Để xây dựng các chiến dịch tiếp thị tốt hơn
▪ Giữ chân nhiều khách hàng hơn: ví dụ với những khách
hàng mua hàng nhiều nhất của công ty sẽ tạo ra các chính
sách riêng cho họ hoặc thu hút lại những người đã mua
hàng trong một khoảng thời gian.
▪ Cải tiến dịch vụ: hiểu rõ khách hàng cho phép bạn điều
chỉnh và tối ưu hóa các dịch vụ của mình để đáp ứng tốt
hơn nhu cầu và mong đợi của khách hàng => giúp cải
thiện sự hài lòng của khách hàng.
▪ Tăng khả năng mở rộng: giúp doanh nghiệp có thể hiểu rõ
hơn về những điều mà khách hàng có thể quan tâm =>
thúc đẩy mở rộng các sản phẩm và dịch vụ mới phù hợp
với đối tượng mục tiêu của họ.

▪Tối ưu hóa giá: Việc có thể xác định tình trạng xã hội và tài
chính của khách hàng giúp doanh nghiệp dễ dàng xác định
giá cả phù hợp cho sản phẩm hoặc dịch vụ của mình mà
khách hàng sẽ cho là hợp lý.
▪ Tăng doanh thu: Dành ít thời gian, nguồn lực và nỗ lực tiếp
thị cho các phân khúc khách hàng có lợi nhuận thấp hơn,
và ngược lại. => hầu hết các phân khúc khách hàng thành
công cuối cùng đều dẫn đến tăng doanh thu và lợi nhuận
cũng như giảm chi phí bán hàng.
    
    """)
    st.image("Customer-Segmentation.png", caption="Customer Segmentation", use_column_width=True)

elif choice == 'Data Understanding':    
  df = pd.read_csv('data.csv', encoding="ISO-8859-1")
  st.write("### Data Overview")
  st.write("shape of data:", df.shape)
  st.write("Columns of data:",df.columns)
  st.write("First five rows of the data:")
  st.write(df.head())
  st.image("info.JPG",caption='infomation')
  st.image("describe.JPG",caption='describe')
  st.image("CountCountry.JPG",caption='Count Country')

elif choice == 'Data preparation': 
  st.write("### Data Cleaning")
    
  
  st.write("Number of missing values:")
  st.write(df.isnull().sum())

  st.write("Number of unique values for each column:")
  st.write(df.nunique())

  st.write("### Dua ve RFM")
  st.write(df_now.head())
  st.write("## Recency (R): đo lường số ngày kể từ lần mua hàng cuối cùng lần truy cập gần đây nhất đến ngày giả định chung để tính toán ví dụ: ngày hiện tại, hoặc ngày max trong danh sách giao dịch")
  st.write("## Frequency (F): đo lường số lượng giao dịch tổng số lần mua hàng được thực hiện trong thời gian nghiên cứu.")
  st.write("## Monetary(M): đo lường số tiền mà mỗi khách hàng chi tiêu trong thời gian nghiên cứu.")
  st.image("rencency.jpg",caption="Distribution of Recency",use_column_width=True)
  st.image("Frequancy.jpg",caption="Distribution of Frequancy",use_column_width=True)
  st.image("Monetary.jpg",caption="Distribution of Monetary",use_column_width=True)




elif choice == 'Modeling & Evaluation':
  st.write("### Modeling with RFM")

  r_labels=range(4,0,-1)
  f_labels=range(1,5)
  m_labels=range(1,5)
  r_groups=pd.qcut(df_now['Recency'].rank(method='first'),q=4,labels=r_labels)
  f_groups=pd.qcut(df_now['Frequency'].rank(method='first'),q=4,labels=f_labels)
  m_groups=pd.qcut(df_now['Monetary'].rank(method='first'),q=4,labels=m_labels)
  df_now=df_now.assign(R=r_groups.values,F=f_groups.values,M=m_groups.values)
  def join_rfm(x): return str(int(x['R']))+str(int(x['F']))+str(int(x['M']))
  df_now['RFM_Segment']=df_now.apply(join_rfm,axis=1)

  df_now['RFM_score']=df_now[['R','F','M']].sum(axis=1)
 

  df_now['RFM_Level']=df_now.apply(rfm_level,axis=1)
  st.write(df_now.head())
  st.image('Coutcustomer.jpg',caption='Count Customer')
  st.image('percentcustomer.jpg',caption='Percent Customer')
  st.image('Customersegments.jpg',caption='Customer Segments')
  st.image('Scatter.jpg')


elif choice=='Predict':

  st.write("### Prediction")

  r_labels=range(4,0,-1)
  f_labels=range(1,5)
  m_labels=range(1,5)
  r_groups=pd.qcut(df_now['Recency'].rank(method='first'),q=4,labels=r_labels)
  f_groups=pd.qcut(df_now['Frequency'].rank(method='first'),q=4,labels=f_labels)
  m_groups=pd.qcut(df_now['Monetary'].rank(method='first'),q=4,labels=m_labels)
  df_now=df_now.assign(R=r_groups.values,F=f_groups.values,M=m_groups.values)
  def join_rfm(x): return str(int(x['R']))+str(int(x['F']))+str(int(x['M']))
  df_now['RFM_Segment']=df_now.apply(join_rfm,axis=1)

  df_now['RFM_score']=df_now[['R','F','M']].sum(axis=1)
 

  df_now['RFM_Level']=df_now.apply(rfm_level,axis=1)

  










  st.write(df_ex)

  st.write("##### 1. Chọn cách nhập thông tin khách hàng")
  type = st.radio("Chọn cách nhập thông tin khách hàng", options=["Nhập mã khách hàng", 
                                                                    "Nhập thông tin khách hàng vào dataframe","Upload file format R F M"])
  if type == "Nhập mã khách hàng":
        st.subheader("Nhập mã khách hàng")
        customer_id = st.text_input("Nhập mã khách hàng")
        st.write("Mã khách hàng:", customer_id)
        st.write("Phân cụm khách hàng...")
    
       
        
        frequency, monetary, recency,level = get_customer_rfm(df_now,customer_id)
        st.write(f"Frequency: {frequency}")
        st.write(f"Monetary: {monetary}")
        st.write(f"Recency: {recency}")
        st.write(f"level: {level}")

        
  elif type =="Nhập thông tin khách hàng vào dataframe":
        st.write("##### 2. Thông tin khách hàng")
        st.write("Nhập thông tin khách hàng")
        df_customer = pd.DataFrame(columns=["R", "F", "M"])
        for i in range(5):
            st.write(f"Khách hàng {i+1}")
            recency = st.slider("R", 1,4, key=f"R_{i}")
            frequency = st.slider("F", 1,4, key=f"F_{i}")
            monetary = st.slider("M", 1,4, key=f"M_{i}")
            df_customer = pd.concat([df_customer, pd.DataFrame({"R": recency, "F": frequency, "M": monetary}, index=[0])], ignore_index=True)
            
            

            


        st.write("##### 3. Phân cụm khách hàng")
        st.write(df_customer)
        st.write("Phân cụm khách hàng...")
        df_customer['RFM_Level']=df_customer.apply(rfm_level,axis=1)
        st.write(df_customer)

  else:

    st.subheader("Select data")
    uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
    df_predict = pd.read_csv(uploaded_file_1,  encoding="ISO-8859-1")
    st.write(df_predict)
    df_predict['RFM_Level']=df_predict.apply(rfm_level,axis=1)
    st.write(df_predict)



  
  



















  






 

  

