from surprise import accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_collaborative_model(model, data):
    """
    حساب مقاييس الخطأ لنموذج SVD
    """
    # 1. تقسيم البيانات لتدريب وتست (80/20)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # 2. التدريب على الجزء الخاص بالتدريب
    model.fit(trainset)
    
    # 3. عمل تنبؤات على جزء الاختبار
    predictions = model.test(testset)
    
    # 4. حساب المقاييس
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    print(f"📊 RMSE: {rmse:.4f}")
    print(f"📊 MAE: {mae:.4f}")
    
    return rmse, mae

def calculate_precision_recall(predictions, threshold=3.5):
    """
    حساب الدقة والاستدعاء (Precision and Recall)
    threshold: التقييم الذي نعتبر فوقه أن الفيلم 'أعجب' المستخدم فعلاً
    """
    from collections import defaultdict

    # تنظيم التوقعات لكل مستخدم
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        # ترتيب الأفلام حسب التقييم المتوقع (الأعلى أولاً)
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # عدد الأفلام التي أعجبت المستخدم فعلاً (الحقيقية >= threshold)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        
        # عدد الأفلام التي توقع النظام أنها ستعجب المستخدم (المتوقعة >= threshold)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:10])
        
        # عدد الأفلام التي توقعها النظام وأعجبت المستخدم فعلاً (True Positives)
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:10])

        # Precision = (الأفلام التي أعجبته من ضمن التي اقترحناها) / (كل التي اقترحناها)
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall = (الأفلام التي أعجبته من ضمن التي اقترحناها) / (كل الأفلام التي تعجبه فعلاً)
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls)
    
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) != 0 else 0

    return avg_precision, avg_recall, f1_score