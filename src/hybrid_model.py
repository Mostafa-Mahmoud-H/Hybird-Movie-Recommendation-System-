import pandas as pd
from src.content_based import build_content_engine
from src.collaborative import train_collaborative_model

class HybridRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
        # 1. تجهيز المحركات عند تشغيل الكلاس
        print("جاري بناء محرك المحتوى...")
        self.cosine_sim = build_content_engine(self.movies_df)
        
        print("جاري تدريب المحرك التعاوني (SVD)...")
        self.svd_model = train_collaborative_model(self.ratings_df)

    def get_hybrid_recommendations(self, user_id, movie_title, top_n=10):
        # أ. الحصول على درجات التشابه من محرك المحتوى (Content-based)
        # نأخذ الأفلام المشابهة للفيلم الذي أدخله المستخدم
        idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # ب. دمج درجات التشابه مع توقعات الـ SVD (Collaborative)
        hybrid_scores = []
        for i, sim_score in sim_scores:
            movie_id = self.movies_df.iloc[i]['movieId']
            
            # توقع التقييم لهذا المستخدم لهذا الفيلم باستخدام SVD
            predicted_rating = self.svd_model.predict(user_id, movie_id).est
            
            # المعادلة الهجينة:
            # سنقوم بتطبيع (Normalize) الـ sim_score ليصبح بين 1 و 5 (مثل التقييمات) ليكون الدمج عادلاً
            # تقييم هجين = (التشابه * وزن) + (التوقع * وزن)
            # سنعطي 50% لكل منهما كمثال
            final_score = (sim_score * 5) * 0.5 + (predicted_rating * 0.5)
            
            hybrid_scores.append((i, final_score))

        # ج. الترتيب واختيار الأفضل
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        
        # تخطي الفيلم نفسه
        movie_indices = [i[0] for i in hybrid_scores if self.movies_df.iloc[i[0]]['title'] != movie_title][:top_n]
        
        return self.movies_df.iloc[movie_indices][['title', 'genres']]