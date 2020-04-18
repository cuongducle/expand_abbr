Môi trường: Python 3.6.9

Link model:https://drive.google.com/file/d/1AkUc0dMOsokeU8kc46rde44v3rJBYSVQ/view?usp=sharing

Tải mô hình cùng chỗ với thư mục, chạy file val_teacher.py để kiểm tra.

Chạy api:FLASK_ENV=development FLASK_APP=expand_api.py flask run
Test api:curl -X POST -H "Content-Type: application/json" -d '{"mytext":"tôi là người ~ vn # ."}' http://localhost:5000/expand
