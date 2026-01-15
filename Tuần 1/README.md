# 📊 Lab 1: Phân Tích & Xử Lý Dữ Liệu Xét Tuyển Đại Học

**Môn học:** Nhập môn Phân tích Dữ liệu và Học sâu  
**Công cụ thực hiện:** Python, Pandas

## 📝 Giới thiệu
Dự án này thực hiện các thao tác tiền xử lý dữ liệu (Data Preprocessing), làm sạch dữ liệu và tạo các đặc trưng mới (Feature Engineering) từ tập dữ liệu điểm thi của học sinh. Mục tiêu là chuẩn bị một bộ dữ liệu sạch, giàu thông tin để phục vụ cho các bước phân tích và mô hình hóa sau này.

## 📂 Dữ liệu đầu vào
* **File:** `dulieuxettuyendaihoc.csv`.
* **Mô tả:** Chứa thông tin điểm số các môn (Toán, Lý, Hóa, Sinh, Văn, Sử, Địa, Ngoại ngữ) của ba năm lớp 10, 11, 12 và điểm thi đại học của 100 học sinh.

## 🛠️ Các bước thực hiện (Workflow)

### 1. Xử lý dữ liệu thiếu (Missing Values)
* **Vấn đề:** Một số cột điểm số bị thiếu giá trị (`NaN`).
* **Giải pháp:** Sử dụng phương pháp thay thế bằng **giá trị trung bình (Mean)** của chính cột đó để đảm bảo tính toàn vẹn dữ liệu cho các bước tính toán sau.
* **Phạm vi:** Áp dụng cho tất cả các biến điểm số từ lớp 10, 11, 12 và điểm thi đại học (`DH1`, `DH2`, `DH3`).

### 2. Tạo biến Trung bình môn (Feature Engineering)
* **Mục tiêu:** Tính điểm trung bình năm cho lớp 10, 11 và 12.
* **Công thức:** Áp dụng công thức trọng số (Toán và Văn hệ số 2):
    $$TBM = \frac{(Toán \times 2 + Văn \times 2 + Các môn khác)}{10}$$
* **Kết quả:** Tạo ra 3 cột mới `TBM1` (Lớp 10), `TBM2` (Lớp 11), `TBM3` (Lớp 12).

### 3. Xếp loại học lực
* **Mục tiêu:** Phân loại học lực dựa trên điểm trung bình (TBM).
* **Quy tắc xếp loại:**
    * `< 5.0`: Yếu (Y)
    * `5.0 - 6.5`: Trung bình (TB)
    * `6.5 - 8.0`: Khá (K)
    * `8.0 - 9.0`: Giỏi (G)
    * `>= 9.0`: Xuất sắc (XS)
* **Kết quả:** Tạo ra 3 cột biến định tính: `XL1`, `XL2`, `XL3`.

### 4. Chuyển đổi thang điểm (Min-Max Normalization)
* **Mục tiêu:** Chuyển đổi điểm TBM từ thang điểm 10 (Việt Nam) sang thang điểm 4 (Mỹ).
* **Phương pháp:** Min-Max Normalization.
* **Công thức:** $Điểm\_Hệ\_4 = Điểm\_Hệ\_10 \times 0.4$.
* **Kết quả:** Tạo ra 3 cột mới: `US_TBM1`, `US_TBM2`, `US_TBM3`.

### 5. Xác định Kết quả Xét tuyển (KQXT)
* **Mục tiêu:** Dự đoán kết quả Đậu/Rớt dựa trên khối thi.
* **Logic xử lý:**
    * **Khối A, A1:** $(DH1 \times 2 + DH2 + DH3) / 4$
    * **Khối B:** $(DH1 + DH2 \times 2 + DH3) / 4$
    * **Khối khác:** $(DH1 + DH2 + DH3) / 3$
* **Điều kiện:** Nếu điểm tổng kết $\geq 5.0$ là Đậu (1), ngược lại là Rớt (0).
* **Kết quả:** Tạo biến `KQXT`.

## 🚀 Kết quả đạt được (Output)
* File dữ liệu đã qua xử lý được lưu trữ thành công với tên: **`processed_dulieuxettuyendaihoc.csv`**.
* File này không còn giá trị rỗng và đã bao gồm đầy đủ các trường thông tin thống kê cần thiết cho việc phân tích sâu hơn (Lab 2).

## 💻 Hướng dẫn chạy (How to run)
1.  Cài đặt thư viện: `pip install pandas`
2.  Đặt file `dulieuxettuyendaihoc.csv` cùng thư mục với script.
3.  Chạy file script Python.
4.  Kiểm tra file kết quả `processed_dulieuxettuyendaihoc.csv` được tạo ra.

---
*Created by Thế Anh - 2026*
