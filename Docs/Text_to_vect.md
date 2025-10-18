#  BIỂU DIỄN TEXT SANG VECTOR

## I. MỤC ĐÍCH VÀ BỐI CẢNH ỨNG DỤNG

Mục tiêu chính của việc biểu diễn text sang vector là để máy tính có thể xử lý và hiểu được dữ liệu văn bản.

*   **Bài toán cơ bản:** Phân loại review tích cực (positive) hay tiêu cực (negative).
*   **Kết quả đầu ra:** Sau khi xử lý, dữ liệu được đưa qua các lớp mạng và kết thúc bằng hàm kích hoạt Sigmoid để đưa ra dự đoán phân loại. Hàm Sigmoid được sử dụng trong bài toán phân loại này.
*   **Ví dụ ứng dụng:** Phân loại review trên tập dữ liệu IMDB.

## II. CÁC BƯỚC TIỀN XỬ LÝ (PREPROCESSING)

### 1. Tokenization (Tách từ)
Việc đầu tiên là tách dữ liệu câu thành các **từ riêng biệt** (các tokens).

### 2. Xây dựng Tập Từ Vựng (Vocabulary)
Sau khi tách từ, tiến hành xây dựng một tập từ điển chứa các từ riêng biệt đã xuất hiện trong toàn bộ dữ liệu (corpus). Kích thước của tập từ điển (vocabulary size) có thể nhỏ (ví dụ 9 từ) hoặc rất lớn (ví dụ 10.000 từ).

## III. CÁC PHƯƠNG PHÁP BIỂU DIỄN CƠ BẢN (BAG OF WORDS - BOW)

### 1. Binary Bag of Words (BOW Nhị Phân)
Phương pháp này biểu diễn câu bằng một vector dựa trên sự **xuất hiện** của các từ trong tập từ vựng.
*   Nếu từ có trong câu, giá trị tương ứng trong vector là **1**.
*   Nếu từ không có trong câu, giá trị là **0**.
*   Vector biểu diễn sẽ có độ dài bằng kích thước của tập từ vựng (ví dụ: 9 chiều).

### 2. Frequency Bag of Words (BOW Tần suất)
Phương pháp này biểu diễn câu bằng vector cho biết **tần suất xuất hiện** (số lần xuất hiện) của mỗi từ trong câu đó.

## IV. XỬ LÝ CÁC VẤN ĐỀ THỰC TẾ

### 1. Vấn đề Vector Thưa (Sparse Vector)
Khi tập từ điển quá lớn (ví dụ 10.000 từ) nhưng câu lại ngắn (ví dụ chỉ 100 từ), vector biểu diễn sẽ có kích thước rất lớn (10.000 chiều) nhưng phần lớn các giá trị là 0, dẫn đến hiện tượng **vector thưa**.

### 2. Giới hạn Kích thước Tập Từ Vựng
Để giải quyết vấn đề vector thưa, có thể **chọn lọc** và giới hạn số lượng từ trong tập từ vựng.
*   Người ta thường sắp xếp các từ theo tần suất xuất hiện từ lớn đến bé và chỉ chọn một số lượng từ nhất định (ví dụ: **5 từ xuất hiện nhiều nhất**) để làm tập từ vựng chính.

### 3. Xử lý Từ Không Có trong Từ Điển (Out-of-Vocabulary - OOV)
Những từ không nằm trong tập từ vựng đã chọn (ví dụ 5 từ xuất hiện nhiều nhất) sẽ được gán là **"out of vocabulary"** (OOV) hay **"unknown"** (UNK).
*   UNK được đưa vào một vị trí riêng biệt và thường được gán một chỉ số cố định (ví dụ: chỉ số 1).

### 4. Đệm Vector (Padding)
Để thống nhất kích thước đầu vào (chiều dài vector) cho mô hình (vì mỗi câu có số lượng từ khác nhau):
*   Thiết lập một **kích thước tối đa** (max length) cố định cho vector.
*   Nếu câu ngắn hơn kích thước tối đa, các vị trí **số 0** sẽ được thêm vào (padding).
*   Padding có thể được thêm vào **phía sau (post)** hoặc **phía trước (pre)**. Ví dụ: Nếu vector cần kích thước là 10 nhưng câu chỉ có 7 từ, thêm 3 số 0 vào sau để đạt kích thước 10.

## V. VECTOR HÓA NÂNG CAO: EMBEDDING LAYER

### 1. Khái niệm và Mục đích
Sau khi biểu diễn câu dưới dạng chỉ số (index) với padding, các chỉ số này được đưa qua **Embedding Layer**.
*   Mục đích là biểu diễn mỗi từ từ một chiều sang **nhiều chiều hơn** (ví dụ 4 chiều hoặc 128 chiều) để mang lại nhiều thông tin và ý nghĩa hơn.
*   Việc này giúp các từ có ý nghĩa tương tự (ví dụ: "dog" và "cat") có **biểu diễn gần nhau** trong không gian vector.

### 2. Cấu trúc Embedding Layer
Layer Embedding có hai tham số chính:
1.  **Input Dimension (Vocabulary Size):** Kích thước của tập từ vựng (ví dụ: 8 hoặc 20 từ).
2.  **Output Dimension (Embedding Dimension):** Số chiều của vector biểu diễn mỗi từ (ví dụ: 4 chiều hoặc 128 chiều).

*   **Tính toán Tham số:** Tổng số tham số (parameters) của Embedding Layer là (Vocabulary size) nhân với (Embedding dimension). Ví dụ: 20 từ x 128 chiều = 2.560 tham số.
*   **Đầu ra:** Đầu ra của Embedding Layer là một ma trận, trong đó mỗi hàng là vector biểu diễn 1 từ. Ví dụ: Với câu dài 500 từ và mỗi từ được biểu diễn bằng vector 128 chiều, đầu ra sẽ có kích thước là **500 x 128**.

## VI. ÁP DỤNG TRONG MÔ HÌNH PHÂN LOẠI (CLASSIFICATION)

### 1. Xây dựng Mô hình (Ví dụ IMDB)
*   **Thiết lập:** Vocabulary size là 10.000, Max length (độ dài câu tối đa) là 500, Embedding dimension là 128.
*   **Dữ liệu:** Tải tập IMDB (25.000 câu review cho tập training và 25.000 câu cho tập validation).
*   **Các tầng mạng:** Sau khi qua Embedding Layer (đầu ra 500 x 128), dữ liệu được **Flatten** (duỗi ra), sau đó đi qua hai tầng Fully Connected, và cuối cùng là lớp đầu ra có **một node** với hàm kích hoạt **Sigmoid**.

### 2. Kết quả và Dự đoán
*   Mô hình có thể đạt được độ chính xác (accuracy) khoảng 85% trên tập validation.
*   **Dự đoán:** Đầu ra của hàm Sigmoid sẽ là một số trong khoảng. Nếu giá trị lớn hơn **0.5**, câu đó được dự đoán là tích cực (positive). Ví dụ: nếu đầu ra là 0.99, câu được phân loại là tích cực.
