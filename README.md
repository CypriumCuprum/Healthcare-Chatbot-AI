🤖 HealthChatBot – Chatbot AI hỏi đáp sức khỏe
Dự án xây dựng một chatbot AI hỗ trợ người dùng nhập các chỉ số sức khỏe cơ bản để chẩn đoán sơ bộ các triệu chứng như ho, cúm, covid, đau bụng, v.v..., đồng thời tích hợp khả năng hỏi đáp sức khỏe qua Gemini (Google Generative AI).

📌 Tính năng
🧾 giao diện cửa sổ chat, trong đó form người dùng chỉ nhập thông tin sức khỏe: Người dùng điền các triệu chứng như:

Ho (0/1)

Sốt (0/1)

Đau họng (0/1)

Đau bụng (0/1)

Mất khứu giác (0/1)

Khó thở (0/1)


🧠 Dự đoán bệnh qua mô hình Deep Learning:

Mô hình tự thiết kế đơn giản bằng Keras/TensorFlow.

Dữ liệu được tự sinh (giả lập) theo logic triệu chứng.

Dự đoán các bệnh thường gặp: Cảm cúm, COVID-19, Viêm họng, Rối loạn tiêu hóa,...

🌐 Chế độ hỏi đáp qua Gemini:

Người dùng có thể đặt câu hỏi về sức khỏe bằng ngôn ngữ tự nhiên.

Câu trả lời được sinh từ Google Gemini API (đảm bảo kiến thức cập nhật, tự nhiên).

python app.py
Ứng dụng sẽ chạy ở http://localhost:5000 (Flask hoặc Streamlit tùy bạn thiết kế giao diện).

🧪 Cấu trúc dự án
src/
│
├── data/
│   └── synthetic_health_data.csv   # Dữ liệu giả lập
│
├── model/
│   └── health_model.h5             # Mô hình DL đã huấn luyện
│   └── train_model.py              # Code huấn luyện
│
├── chatbot/
│   └── gemini_client.py            # Gửi câu hỏi đến Gemini API
│
├── app.py                          # Flask hoặc Streamlit app
├── requirements.txt                # Các thư viện cần thiết
README.md
🧠 Mô hình Deep Learning
Dữ liệu tự sinh: tạo ngẫu nhiên các mẫu triệu chứng và gán nhãn bệnh dựa vào logic y khoa cơ bản.

Mô hình: DNN với 2–3 lớp ẩn, hàm kích hoạt ReLU, đầu ra Softmax.

Đánh giá mô hình: Accuracy > 90% trên tập test giả lập.

🤖 Tích hợp Gemini (Google Generative AI)
Yêu cầu API key từ Google AI Studio.

Hỗ trợ hỏi các câu như:

"Tôi bị ho và sốt 2 ngày nay, có thể là bệnh gì?"

"Làm sao để phòng tránh COVID?"

"Khi nào nên đi khám bác sĩ?"

📷 Giao diện (demo)
(chèn hình ảnh nếu có)

🔒 Ghi chú
Không thay thế chẩn đoán y tế thật sự.

Dữ liệu và mô hình mang tính minh họa.

📄 Giấy phép
MIT License

