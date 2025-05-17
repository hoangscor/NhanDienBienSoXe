🇻🇳 HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE (ANPR) DỰA TRÊN ẢNH — FLASK + OPENCV + TESSERACT

📷 Tự động nhận diện biển số xe từ ảnh
🧠 Tách ký tự, nhận dạng và quản lý trạng thái xe vào/ra
📦 Lưu trữ ảnh xử lý, ảnh gốc và kết quả
🛠️ Giao diện web bằng Flask (index & admin)

🎯 MỤC TIÊU DỰ ÁN

Phát triển một hệ thống nhận dạng biển số xe (ANPR) hoạt động theo thời gian thực hoặc xử lý ảnh thủ công. Hệ thống có khả năng:

Tự động phát hiện và trích xuất biển số xe từ ảnh đầu vào

Xử lý ảnh theo 11 bước pipeline chuyên sâu + 6 bước mô tả logic chính

Tách ký tự và nhận diện biển số bằng OCR (Tesseract)

Quản lý việc xe vào - xe ra, tính phí, lưu trữ thông tin và ảnh

Hỗ trợ chế độ tự động và thủ công 
⚙️ CHỨC NĂNG CHÍNH

Nhận diện biển số xe

Người dùng có thể chọn ảnh mẫu hoặc upload ảnh/chụp webcam

Xử lý qua pipeline 11 bước + hiển thị 6 bước chính để minh họa

Tách ký tự, nhận dạng ký tự và ghép thành biển số hoàn chỉnh

Quản lý xe vào/ra

Khi phát hiện biển số:

Nếu chưa có trong danh sách → Cho phép Xe Vào

Nếu đã có, chưa ra → Cho phép Xe Ra

Lưu thời gian, biển số, phí, ảnh vào/ra vào database

Chế độ tự động

Kích hoạt tự động, hệ thống sẽ liên tục quét thư mục images/

Khi có ảnh mới, tự động nhận diện và cập nhật trạng thái xe

Giao diện admin

Quản lý toàn bộ lịch sử xe vào ra

Hiển thị ảnh vào và ảnh ra

Sửa thông tin, xóa bản ghi, xuất file tùy chỉnh (tùy chọn mở rộng)

🖼️ CÁC BƯỚC XỬ LÝ ẢNH

11 bước pipeline

Resize ảnh về chuẩn

Chuyển grayscale

Cân bằng histogram

Lọc nhiễu (bilateral filter)

Tách biên (Canny)

Tìm contour biển số

Crop vùng biển số

Khử nhiễu + threshold

Nhị phân hóa

Tách và đánh dấu ký tự

Kết quả cuối cùng

6 bước chính minh họa

Phát hiện vị trí biển số

Cắt và hiệu chỉnh vùng biển số

Nhị phân hóa biển số

Phân tách ký tự

OCR từng ký tự

Ghép lại chuỗi biển số
