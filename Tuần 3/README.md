# 🏥 LAB 3: LÀM SẠCH DỮ LIỆU CƠ BẢN

## 📌 Tổng quan
Bài thực hành này tập trung vào kỹ thuật **Data Cleansing** (Làm sạch dữ liệu) sử dụng thư viện Pandas[cite: 237]. [cite_start]Sinh viên sẽ làm việc với tập dữ liệu y khoa về huyết áp và nhịp tim của bệnh nhân (`patient_heart_rate.csv`)[cite: 237, 239], giải quyết các vấn đề thực tế thường gặp như dữ liệu thiếu, sai định dạng, hoặc trùng lặp.

## 🛠 Yêu cầu kỹ thuật
* **Ngôn ngữ:** Python
* **Thư viện chính:** Pandas
* **Dữ liệu đầu vào:** File `patient_heart_rate.csv` (chứa thông tin Id, Name, Age, Weight, Heart Rates...)[cite: 239].

## 📝 Nội dung thực hiện

### 1️⃣ Nhận diện và Xử lý lỗi dữ liệu cơ bản
Sinh viên cần giải quyết lần lượt các vấn đề (problems) sau:
* ⚠️ **Vấn đề 1 (Missing Header):** Tải dữ liệu và bổ sung dòng tiêu đề bị thiếu cho file CSV[cite: 241, 249].
* ⚠️ **Vấn đề 2 (Multiple Variables):** Tách cột `Name` chứa cả Họ và Tên thành 2 cột riêng biệt: `Firstname` và `Lastname`[cite: 242, 260].
* ⚠️ **Vấn đề 3 (Inconsistent Units):** Chuẩn hóa cột `Weight` về cùng đơn vị (chuyển đổi từ `lbs` sang `kgs` và loại bỏ ký tự thừa)[cite: 266, 267].
* ⚠️ **Vấn đề 4 (Empty Rows):** Phát hiện và xóa các dòng dữ liệu rỗng (NaN)[cite: 284].

### 2️⃣ Xử lý dữ liệu nâng cao
* 🔄 **Vấn đề 5 (Duplicates):** Xử lý các dòng dữ liệu bị trùng lặp thông tin (dựa trên `Firstname`, `Lastname`, `Age`, `Weight`)[cite: 290, 291].
* 🔡 **Vấn đề 6 (Non-ASCII):** Loại bỏ các ký tự lỗi font, không phải bảng mã ASCII trong tên[cite: 293].
* 🧩 **Vấn đề 7 (Missing Values - Age & Weight):** Thống kê dữ liệu thiếu và xử lý theo quy tắc:
    * Nếu có dữ liệu ở một trong hai cột, điền giá trị thiếu bằng **Mean** (giá trị trung bình).
    * Nếu thiếu cả hai, xóa dòng dữ liệu đó[cite: 310, 311, 312].

### 3️⃣ Tái cấu trúc dữ liệu (Data Reshaping)
* 📉 **Vấn đề 8 (Column Decomposition & Melting):**
    * Phân rã các cột chứa thông tin gộp (ví dụ: `m0006`, `m0612`...) thành các cột `PulseRate`, `Sex` (giới tính) và `Time` (khoảng thời gian)[cite: 313, 316].
    * Sử dụng kỹ thuật `melt` để chuyển đổi dữ liệu từ dạng rộng (wide) sang dạng dài (long)[cite: 323].

### 4️⃣ Xử lý dữ liệu thiếu phức tạp (Imputation Logic)
* 🩺 **Khảo sát & Điền khuyết Huyết áp:** Thực hiện quy trình xử lý ưu tiên theo thứ tự cho dữ liệu huyết áp bị thiếu:
    1. Trung bình cộng của giá trị liền trước và liền sau.
    2. Trung bình 2 giá trị liền trước.
    3. Trung bình 2 giá trị liền sau.
    4. Trung bình của chính người đó.
    5. Trung bình của nhóm giới tính hoặc toàn bộ dữ liệu[cite: 337, 338, 339, 341, 343, 345].

### 5️⃣ Lưu trữ kết quả
* Rút gọn, `reindex` lại dữ liệu và lưu thành file hoàn chỉnh: `patient_heart_rate_clean.csv`[cite: 347, 348].