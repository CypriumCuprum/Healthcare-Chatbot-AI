import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Union

class HealthPredictor:
    def __init__(self):
        """Initialize the health predictor with the trained model."""
        self.model_path = os.path.join(os.path.dirname(__file__), 'health_model.h5')
        self.class_names_path = os.path.join(os.path.dirname(__file__), 'class_names.npy')
        
        # Load the model if it exists
        self.model = None
        self.class_names = None
        self.feature_names = ['ho', 'sot', 'dau_hong', 'dau_bung', 'mat_khuu_giac', 'kho_tho']
        
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model and class names.
        
        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.class_names_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.class_names = np.load(self.class_names_path, allow_pickle=True)
                return True
            else:
                print("Model files not found. Please train the model first.")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded.
        
        Returns:
            bool: True if the model is loaded, False otherwise.
        """
        return self.model is not None and self.class_names is not None
    
    def predict(self, symptoms: Dict[str, int]) -> Tuple[str, Dict[str, float]]:
        """Predict the disease based on the given symptoms.
        
        Args:
            symptoms: A dictionary of symptoms (feature_name -> 0/1).
                Example: {'ho': 1, 'sot': 1, 'dau_hong': 0, 'dau_bung': 0, 'mat_khuu_giac': 0, 'kho_tho': 0}
            
        Returns:
            Tuple containing:
                - The predicted disease name
                - A dictionary of prediction probabilities for each disease
        """
        if not self.is_model_loaded():
            return "Model not loaded", {}
        
        # Prepare the input data
        X = np.zeros(len(self.feature_names))
        for i, feature in enumerate(self.feature_names):
            X[i] = symptoms.get(feature, 0)
        
        # Reshape for model input (batch size of 1)
        X = X.reshape(1, -1)
        
        # Make prediction
        predictions = self.model.predict(X)[0]
        
        # Get the predicted class and probabilities
        predicted_class_index = np.argmax(predictions)
        predicted_disease = self.class_names[predicted_class_index]
        
        # Create a dictionary of disease -> probability
        probabilities = {
            disease: float(predictions[i]) 
            for i, disease in enumerate(self.class_names)
        }
        
        return predicted_disease, probabilities
    
    def get_disease_description(self, disease: str) -> str:
        """Get a description of the disease.
        
        Args:
            disease: The name of the disease.
            
        Returns:
            str: A description of the disease.
        """
        descriptions = {
            'cam_cum': "Cảm cúm: Bệnh nhiễm trùng đường hô hấp gây ra bởi virus cúm. Các triệu chứng thường gặp là ho, sốt, đau họng, đau nhức cơ thể.",
            'covid_19': "COVID-19: Bệnh nhiễm trùng đường hô hấp do virus SARS-CoV-2 gây ra. Các triệu chứng phổ biến là sốt, ho, mất khứu giác, mệt mỏi và khó thở.",
            'viem_hong': "Viêm họng: Tình trạng viêm và kích ứng ở họng, thường gây cảm giác đau khi nuốt. Có thể do virus hoặc vi khuẩn gây ra.",
            'roi_loan_tieu_hoa': "Rối loạn tiêu hóa: Các vấn đề liên quan đến hệ tiêu hóa như đau bụng, buồn nôn, tiêu chảy hoặc táo bón.",
            'binh_thuong': "Bình thường: Không có dấu hiệu của bệnh nghiêm trọng. Bạn có thể đang gặp các triệu chứng nhẹ hoặc tạm thời."
        }
        
        return descriptions.get(disease, f"Không có thông tin về {disease}")
    
    def get_recommendations(self, disease: str) -> List[str]:
        """Get health recommendations based on the predicted disease.
        
        Args:
            disease: The name of the disease.
            
        Returns:
            List[str]: A list of recommendations.
        """
        recommendations = {
            'cam_cum': [
                "Nghỉ ngơi đầy đủ và uống nhiều nước",
                "Sử dụng thuốc hạ sốt như paracetamol nếu cần",
                "Dùng thuốc giảm ho và đau họng theo chỉ dẫn",
                "Tránh tiếp xúc gần với người khác để tránh lây nhiễm"
            ],
            'covid_19': [
                "Cách ly ngay lập tức và làm xét nghiệm COVID-19",
                "Theo dõi các triệu chứng và liên hệ bác sĩ nếu tình trạng xấu đi",
                "Nghỉ ngơi, uống nhiều nước và dùng thuốc hạ sốt nếu cần",
                "Thực hiện các biện pháp phòng ngừa để tránh lây nhiễm cho người khác"
            ],
            'viem_hong': [
                "Súc họng bằng nước muối ấm",
                "Uống nhiều nước ấm và nghỉ ngơi giọng nói",
                "Dùng thuốc giảm đau họng theo chỉ dẫn",
                "Nếu triệu chứng kéo dài trên 5 ngày, hãy đi khám bác sĩ"
            ],
            'roi_loan_tieu_hoa': [
                "Uống nhiều nước để tránh mất nước",
                "Ăn các thực phẩm dễ tiêu như cơm, bánh mì trắng, chuối",
                "Tránh thực phẩm cay, nhiều dầu mỡ, caffeine và đồ uống có cồn",
                "Nếu triệu chứng kéo dài hoặc nặng, hãy đi khám bác sĩ"
            ],
            'binh_thuong': [
                "Duy trì lối sống lành mạnh với chế độ ăn cân bằng",
                "Tập thể dục đều đặn",
                "Ngủ đủ giấc (7-8 giờ mỗi đêm)",
                "Uống đủ nước (khoảng 2 lít mỗi ngày)"
            ]
        }
        
        return recommendations.get(disease, ["Vui lòng tham khảo ý kiến bác sĩ để được tư vấn cụ thể."]) 