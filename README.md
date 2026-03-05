# Dự án Dự đoán Giá Nhà Boston với XGBoost

## 📌 Giới thiệu

Dự án này xây dựng một mô hình Machine Learning sử dụng thuật toán **XGBoost (Extreme Gradient Boosting)** để dự đoán giá nhà tại Boston dựa trên các đặc điểm của khu vực và bất động sản.

## 📊 Dataset

**Boston Housing Dataset** bao gồm:
- **506 mẫu dữ liệu**
- **13 đặc trưng đầu vào**
- **1 biến mục tiêu** (giá nhà)

### Các đặc trưng (Features):

| Tên | Mô tả |
|-----|-------|
| `CRIM` | Tỷ lệ tội phạm trên đầu người theo thị trấn |
| `ZN` | Tỷ lệ đất dân cư được phân vùng cho các lô trên 25,000 sq.ft |
| `INDUS` | Tỷ lệ diện tích kinh doanh phi bán lẻ trên thị trấn |
| `CHAS` | Biến giả sông Charles (1 nếu giáp sông; 0 nếu không) |
| `NOX` | Nồng độ oxit nitric (phần trên 10 triệu) |
| `RM` | Số phòng trung bình trên mỗi căn hộ |
| `AGE` | Tỷ lệ các đơn vị nhà được xây dựng trước năm 1940 |
| `DIS` | Khoảng cách có trọng số đến 5 trung tâm việc làm tại Boston |
| `RAD` | Chỉ số khả năng tiếp cận đường cao tốc xuyên tâm |
| `TAX` | Thuế suất tài sản đầy đủ trên $10,000 |
| `PTRATIO` | Tỷ lệ học sinh-giáo viên theo thị trấn |
| `B` | Tỷ lệ người da đen theo thị trấn |
| `LSTAT` | Phần trăm dân số thuộc tầng lớp thấp |

### Biến mục tiêu:
- **MEDV**: Giá trị trung vị của các ngôi nhà do chủ sở hữu ở (tính bằng $1000)


## 🔧 Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- Jupyter Notebook hoặc JupyterLab

### Cài đặt thư viện

```bash
pip install numpy pandas xgboost matplotlib scikit-learn
```

**Hoặc** tạo file `requirements.txt`:

```text
numpy>=1.21.0
pandas>=1.3.0
xgboost>=1.5.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

Sau đó chạy:
```bash
pip install -r requirements.txt
```

## 🚀 Cách sử dụng

### 1. Mở thư mục dự án

```bash
cd e:\KhaiPhaDuLieu\thuyettrinh
```

### 2. Mở Jupyter Notebook

```bash
jupyter notebook XGBoost_BostonHousing.ipynb
```

### 3. Chạy các cell theo thứ tự

Notebook được chia thành các phần rõ ràng:

1. ✅ **Cài đặt XGBoost** - Cài đặt thư viện cần thiết
2. ✅ **Nhập thư viện** - Import các thư viện cần dùng
3. ✅ **Tải dữ liệu** - Tải dataset Boston Housing từ OpenML
4. ✅ **Mô tả thống kê** - Phân tích tổng quan dữ liệu
5. ✅ **Khởi tạo mô hình** - Tạo mô hình XGBRegressor
6. ✅ **Huấn luyện** - Train mô hình trên dữ liệu
7. ✅ **Đánh giá** - Tính toán các metrics (MSE, RMSE, R²)
8. ✅ **Trực quan hóa** - Vẽ biểu đồ Feature Importance và Actual vs Predicted

## 🎯 Quy trình xây dựng mô hình

### 1. **Chuẩn bị dữ liệu**
```python
boston = fetch_openml(name="boston", version=1)
X = boston.data
y = boston.target.astype(float)

# Chia dữ liệu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 2. **Cấu hình mô hình XGBoost**

```python
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Hàm mất mát: Sai số bình phương
    n_estimators=100,              # Số cây quyết định
    max_depth=4,                   # Độ sâu tối đa của cây
    learning_rate=0.1,             # Tốc độ học
    reg_lambda=1,                  # Tham số chính quy hóa L2
    random_state=42,               # Seed cho tính tái lập
    enable_categorical=True        # Hỗ trợ biến phân loại
)
```

### 3. **Huấn luyện và đánh giá**

```python
# Huấn luyện
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

Các chỉ số đánh giá:
- **MSE (Mean Squared Error)**: Trung bình sai số bình phương
- **RMSE (Root Mean Squared Error)**: Căn bậc hai của MSE
- **R² Score**: Hệ số xác định (0-1), càng cao càng tốt

### 4. **Trực quan hóa**

- **Feature Importance**: Đánh giá tầm quan trọng của từng đặc trưng
- **Actual vs Predicted**: So sánh giá thực tế và dự đoán

## 📈 Kết quả mong đợi

Với cấu hình mặc định, mô hình thường đạt:
- ✅ **R² Score**: > 0.85
- ✅ **RMSE**: < 5.0
- ✅ **MSE**: < 25.0

### Các đặc trưng quan trọng nhất (thường thấy):
1. 🏠 **RM** (Số phòng) - Ảnh hưởng mạnh nhất đến giá nhà
2. 📊 **LSTAT** (% dân số tầng lớp thấp) - Tương quan nghịch với giá
3. 🚗 **DIS** (Khoảng cách đến trung tâm việc làm)
4. 🏭 **INDUS** (Tỷ lệ diện tích kinh doanh)

## 📚 Kiến thức về XGBoost

### XGBoost là gì?
XGBoost (Extreme Gradient Boosting) là một thuật toán machine learning mạnh mẽ dựa trên:
- **Gradient Boosting**: Xây dựng mô hình theo cách tuần tự, mỗi cây học từ lỗi của cây trước
- **Decision Trees**: Sử dụng cây quyết định làm base learner
- **Regularization**: Giảm overfitting thông qua L1/L2 regularization

### Ưu điểm của XGBoost:
- ⚡ **Tốc độ cao**: Tối ưu hóa tính toán và song song hóa
- 🎯 **Hiệu suất tốt**: Thường đạt kết quả cao trong competitions
- 🛡️ **Chống overfitting**: Có nhiều kỹ thuật regularization
- 📊 **Xử lý missing values**: Tự động xử lý dữ liệu thiếu
- 🔧 **Linh hoạt**: Nhiều tùy chỉnh hyperparameters

### Các tham số quan trọng:
- `n_estimators`: Số lượng cây (trees)
- `max_depth`: Độ sâu tối đa của mỗi cây
- `learning_rate`: Tốc độ học (learning rate)
- `reg_lambda`: Regularization L2
- `reg_alpha`: Regularization L1

## 🔍 Mở rộng dự án

Một số hướng phát triển:

### 1. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 2. Feature Engineering
- Tạo features tương tác (ví dụ: RM * DIS)
- Chuẩn hóa dữ liệu (StandardScaler, MinMaxScaler)
- Biến đổi logarit cho các features lệch phân phối

### 3. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, 
                         scoring='neg_mean_squared_error')
```

### 4. Model Deployment
- Lưu mô hình: `model.save_model('boston_model.json')`
- Tạo API với Flask/FastAPI
- Containerize với Docker

## 📖 Tài liệu tham khảo

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Parameters Guide](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Boston Housing Dataset - OpenML](https://www.openml.org/d/531)


