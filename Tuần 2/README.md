# 📚 LAB 2: BÀI THỰC HÀNH TRÌNH BÀY DỮ LIỆU

## 📌 Tổng quan
Bài thực hành này tập trung vào kỹ năng **Trình bày và Trực quan hóa dữ liệu** (Data Visualization) dựa trên tập dữ liệu điểm thi đại học đã được xử lý từ Lab 1 (`processed_dulieuxettuyendaihoc.csv`). Mục tiêu là giúp sinh viên hiểu rõ hơn về phân bố dữ liệu, mối tương quan giữa các biến và cách sử dụng các biểu đồ để rút ra thông tin hữu ích.

## 🛠 Yêu cầu kỹ thuật
* **Ngôn ngữ:** Python
* **Thư viện sử dụng:** Pandas, Matplotlib, Seaborn (tùy chọn).
* **Dữ liệu đầu vào:** File `processed_dulieuxettuyendaihoc.csv` (kết quả từ Lab 1).

## 📝 Nội dung thực hiện

### 1️⃣ Phần 1: Thống kê dữ liệu (Statistics)
Sử dụng **Pivot-table** và các hàm thống kê cơ bản để:
* Sắp xếp dữ liệu điểm thi (DH1, DH2).
* Thống kê các chỉ số mô tả: `count`, `sum`, `mean`, `median`, `min`, `max`, `std`, `Q1`, `Q2`, `Q3`.
* Phân tích điểm thi theo các nhóm: Khối thi (KT), Khu vực (KV), và Dân tộc (DT).

### 2️⃣ Phần 2: Trình bày dữ liệu (Data Presentation)
Lập bảng tần số, tần suất và lọc dữ liệu theo điều kiện cụ thể:
* Lập bảng tần số/tần suất cho giới tính (GT).
* Trình bày dữ liệu điểm quy đổi sang thang điểm 4 (US_TBM).
* Lọc và hiển thị dữ liệu theo các điều kiện phức tạp (ví dụ: Học sinh nam, dân tộc Kinh, khu vực 2NT có điểm thi thỏa mãn điều kiện sàn).

### 3️⃣ Phần 3: Trực quan hóa dữ liệu theo nhóm (Categorical Visualization)
Vẽ các biểu đồ để so sánh các nhóm dữ liệu:
* ✅ **Biểu đồ cột (Bar Chart):** So sánh số lượng học sinh Đậu/Rớt theo Khối thi, Khu vực, Dân tộc, Giới tính.
* ✅ **Biểu đồ Unstacked:** Phân loại học sinh nữ theo xếp loại học lực (Yếu, TB, Khá, Giỏi, Xuất sắc).

### 4️⃣ Phần 4: Trực quan hóa dữ liệu nâng cao (Advanced Visualization)
Sử dụng các biểu đồ đường để theo dõi biến động điểm số:
* 📉 **Simple Line Plot:** Biểu diễn điểm Toán học kỳ 1 (T1).
* 📉 **Multiple Line Plot & Drop-line Plot:** So sánh điểm T1 sau khi đã phân lớp (Kém, TB, Khá, Giỏi).

### 5️⃣ Phần 5: Mô tả dữ liệu và khảo sát phân phối (Distribution & Correlation)
Sử dụng các biểu đồ thống kê chuyên sâu để đánh giá độ tin cậy và phân phối của biến:
* 📦 **Box-Plot:** Xác định độ tập trung, phân tán và các giá trị ngoại lai (outliers).
* 📊 **Histogram:** Xem xét hình dáng phân phối (độ lệch Skewness, độ nhọn Kurtosis).
* 📈 **QQ-Plot:** Kiểm định xem dữ liệu có tuân theo phân phối chuẩn hay không.
* 🔗 **Scatter Plot:** Khảo sát tương quan giữa điểm thi Đại học (DH1) và điểm học bạ (T1), hoặc tương quan giữa các môn thi (DH1, DH2, DH3).