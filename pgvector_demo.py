"""
OpenTenbase 产品推荐系统 Demo
基于向量相似度的产品推荐系统, 使用OpenTenbase + pgvector扩展
"""

import os
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import json

class ProductRecommendationSystem:
    def __init__(self, db_config: Dict[str, str]):
        """
        初始化产品推荐系统
        
        Args:
            db_config: 数据库连接配置
        """
        self.db_config = db_config
        self.conn = None
        self.model = SentenceTransformer('./model', local_files_only=True)
        self.embedding_dim = 384  # all-MiniLM-L6-v2 的向量维度
        
    def connect_db(self):
        """连接数据库"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("✅ 数据库连接成功")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            raise
    
    def setup_database(self):
        """设置数据库，创建扩展和表"""
        cur = self.conn.cursor()
        
        # 1. 创建pgvector扩展
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✅ pgvector扩展创建成功")
        except Exception as e:
            print(f"❌ 创建pgvector扩展失败: {e}")
            
        # 2. 创建产品表
        create_products_table = f"""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            category VARCHAR(100),
            price DECIMAL(10,2),
            brand VARCHAR(100),
            tags TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description_embedding VECTOR({self.embedding_dim})
        );
        """
        
        try:
            cur.execute(create_products_table)
            print("✅ 产品表创建成功")
        except Exception as e:
            print(f"❌ 创建产品表失败: {e}")
            
        # 3. 创建用户行为表
        create_user_behavior_table = """
        CREATE TABLE IF NOT EXISTS user_behaviors (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            action_type VARCHAR(50) NOT NULL, -- 'view', 'like', 'purchase', 'add_to_cart'
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            cur.execute(create_user_behavior_table)
            print("✅ 用户行为表创建成功")
        except Exception as e:
            print(f"❌ 创建用户行为表失败: {e}")
            
        self.conn.commit()
        cur.close()
    
    def create_vector_index(self):
        """创建向量索引以提高查询性能"""
        cur = self.conn.cursor()
        
        # 创建HNSW索引（适合高维向量的近似最近邻搜索）
        index_sql = """
        CREATE INDEX IF NOT EXISTS products_description_embedding_idx 
        ON products USING hnsw (description_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
        
        try:
            cur.execute(index_sql)
            self.conn.commit()
            print("✅ 向量索引创建成功")
        except Exception as e:
            print(f"❌ 创建向量索引失败: {e}")
            
        cur.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        为文本生成嵌入向量
        
        Args:
            text: 要编码的文本
            
        Returns:
            嵌入向量列表
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def insert_sample_products(self):
        """插入示例产品数据"""
        sample_products = [
            {
                "name": "iPhone 15 Pro",
                "description": "最新款苹果手机，配备A17 Pro芯片，钛金属设计，三摄像头系统，支持5G网络",
                "category": "手机",
                "price": 7999.00,
                "brand": "Apple",
                "tags": ["智能手机", "5G", "高端", "摄影"]
            },
            {
                "name": "MacBook Air M2",
                "description": "轻薄笔记本电脑，搭载M2芯片，13.6英寸Liquid Retina显示屏，续航18小时",
                "category": "笔记本电脑",
                "price": 8999.00,
                "brand": "Apple",
                "tags": ["笔记本", "轻薄", "长续航", "办公"]
            },
            {
                "name": "Sony WH-1000XM5",
                "description": "旗舰降噪耳机，业界领先的降噪技术，30小时续航，Hi-Res音质认证",
                "category": "耳机",
                "price": 2399.00,
                "brand": "Sony",
                "tags": ["降噪耳机", "无线", "音乐", "通勤"]
            },
            {
                "name": "小米13 Ultra",
                "description": "徕卡影像旗舰手机，1英寸大底主摄，骁龙8 Gen2处理器，67W快充",
                "category": "手机",
                "price": 5999.00,
                "brand": "小米",
                "tags": ["智能手机", "摄影", "快充", "徕卡"]
            },
            {
                "name": "ThinkPad X1 Carbon",
                "description": "商务旗舰笔记本，碳纤维机身，14英寸2.8K屏幕，Intel第12代处理器",
                "category": "笔记本电脑",
                "price": 12999.00,
                "brand": "联想",
                "tags": ["商务", "轻薄", "办公", "ThinkPad"]
            },
            {
                "name": "AirPods Pro 2",
                "description": "苹果无线降噪耳机，H2芯片，自适应降噪，空间音频，MagSafe充电",
                "category": "耳机",
                "price": 1899.00,
                "brand": "Apple",
                "tags": ["无线耳机", "降噪", "苹果生态", "便携"]
            },
            {
                "name": "华为Mate 60 Pro",
                "description": "华为旗舰手机，麒麟9000S处理器，卫星通话功能，超光变摄影系统",
                "category": "手机",
                "price": 6999.00,
                "brand": "华为",
                "tags": ["智能手机", "卫星通话", "摄影", "旗舰"]
            },
            {
                "name": "Dell XPS 13 Plus",
                "description": "超薄商务笔记本，13.4英寸OLED触摸屏，Intel第12代i7处理器，极简设计",
                "category": "笔记本电脑",
                "price": 11999.00,
                "brand": "戴尔",
                "tags": ["超薄", "商务", "OLED", "触摸屏"]
            }
        ]
        
        cur = self.conn.cursor()
        
        for product in sample_products:
            # 生成描述的嵌入向量
            embedding = self.generate_embedding(product["description"])
            
            insert_sql = """
            INSERT INTO products (name, description, category, price, brand, tags, description_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            try:
                cur.execute(insert_sql, (
                    product["name"],
                    product["description"],
                    product["category"],
                    product["price"],
                    product["brand"],
                    product["tags"],
                    embedding
                ))
            except Exception as e:
                print(f"❌ 插入产品 {product['name']} 失败: {e}")
        
        self.conn.commit()
        cur.close()
        print("✅ 示例产品数据插入成功")
    
    def insert_sample_user_behaviors(self):
        """插入示例用户行为数据"""
        sample_behaviors = [
            {"user_id": 1, "product_id": 1, "action_type": "view", "rating": None},
            {"user_id": 1, "product_id": 1, "action_type": "like", "rating": 5},
            {"user_id": 1, "product_id": 2, "action_type": "view", "rating": None},
            {"user_id": 1, "product_id": 3, "action_type": "purchase", "rating": 5},
            {"user_id": 2, "product_id": 4, "action_type": "view", "rating": None},
            {"user_id": 2, "product_id": 4, "action_type": "purchase", "rating": 5},
            {"user_id": 2, "product_id": 6, "action_type": "like", "rating": 4},
            {"user_id": 3, "product_id": 5, "action_type": "view", "rating": None},
            {"user_id": 3, "product_id": 8, "action_type": "add_to_cart", "rating": None},
        ]
        
        cur = self.conn.cursor()
        
        for behavior in sample_behaviors:
            insert_sql = """
            INSERT INTO user_behaviors (user_id, product_id, action_type, rating)
            VALUES (%s, %s, %s, %s)
            """
            
            try:
                cur.execute(insert_sql, (
                    behavior["user_id"],
                    behavior["product_id"],
                    behavior["action_type"],
                    behavior["rating"]
                ))
            except Exception as e:
                print(f"❌ 插入用户行为失败: {e}")
        
        self.conn.commit()
        cur.close()
        print("✅ 示例用户行为数据插入成功")
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Tuple]:
        """
        基于语义相似度搜索产品
        
        Args:
            query: 搜索查询文本
            limit: 返回结果数量
            
        Returns:
            相似产品列表，包含相似度分数
        """
        # 生成查询的嵌入向量
        query_embedding = self.generate_embedding(query)
        
        cur = self.conn.cursor()
        
        # 使用余弦相似度进行向量搜索
        search_sql = """
        SELECT 
            id, name, description, category, price, brand, tags,
            1 - (description_embedding <=> %s::vector) as similarity
        FROM products
        WHERE description_embedding IS NOT NULL
        ORDER BY description_embedding <=> %s::vector
        LIMIT %s;
        """
        
        try:
            cur.execute(search_sql, (query_embedding, query_embedding, limit))
            results = cur.fetchall()
            cur.close()
            return results
        except Exception as e:
            print(f"❌ 语义搜索失败: {e}")
            cur.close()
            return []
    
    def recommend_by_user_history(self, user_id: int, limit: int = 5) -> List[Tuple]:
        """
        基于用户历史行为推荐产品
        """

        cur = self.conn.cursor()

        # 获取用户喜欢的产品
        get_user_preferences_sql = """
        SELECT p.description_embedding, ub.rating
        FROM user_behaviors ub
        JOIN products p ON ub.product_id = p.id
        WHERE ub.user_id = %s 
        AND ub.action_type IN ('like', 'purchase')
        AND p.description_embedding IS NOT NULL
        AND ub.rating >= 3;
        """

        cur.execute(get_user_preferences_sql, (user_id,))
        user_preferences = cur.fetchall()
        
        if not user_preferences:
            cur.close()
            return []
        
        # 计算用户偏好向量（加权平均）
        weighted_embeddings = []
        total_weight = 0
        
        for embedding_bytes, rating in user_preferences:
            # 转为字符串
            if isinstance(embedding_bytes, bytes):
                embedding_str = embedding_bytes.decode('utf-8')
            else:
                embedding_str = embedding_bytes

            embedding = np.array(json.loads(embedding_str))  # 解析 JSON 数组

            weight = rating / 5.0
            weighted_embeddings.append(embedding * weight)
            total_weight += weight
        
        if total_weight == 0:
            cur.close()
            return []
        

        user_preference_vector = np.sum(weighted_embeddings, axis=0) / total_weight

        # 推荐查询（使用 vector 相似度）
        recommend_sql = """
            SELECT 
                p.id, p.name, p.description, p.category, p.price, p.brand, p.tags,
                1 - (p.description_embedding <=> %s::vector) as similarity
            FROM products p
            WHERE p.description_embedding IS NOT NULL
            AND p.id NOT IN (
                SELECT DISTINCT product_id 
                FROM user_behaviors 
                WHERE user_id = %s
            )
            ORDER BY p.description_embedding <=> %s::vector
            LIMIT %s;
            """
        user_preference_vector = user_preference_vector.tolist()
        try:
            # ✅ 此处直接传入 numpy array，pgvector 会自动处理
            cur.execute(recommend_sql, (
                user_preference_vector,  # numpy array，pgvector 能识别
                user_id,
                user_preference_vector,
                limit
            ))
            results = cur.fetchall()
            cur.close()
            return results
        except Exception as e:
            print(f"❌ 个性化推荐失败: {e}")
            cur.close()
            return []
    
    def hybrid_search(self, query: str, category: str = None, 
                     price_range: Tuple[float, float] = None, limit: int = 5) -> List[Tuple]:
        """
        混合搜索：结合语义搜索和传统筛选
        
        Args:
            query: 搜索查询
            category: 产品类别筛选
            price_range: 价格范围筛选 (min_price, max_price)
            limit: 返回结果数量
            
        Returns:
            筛选后的相似产品列表
        """
        query_embedding = self.generate_embedding(query)
        
        # 构建动态SQL查询
        base_sql = """
        SELECT 
            id, name, description, category, price, brand, tags,
            1 - (description_embedding <=> %s::vector) as similarity
        FROM products
        WHERE description_embedding IS NOT NULL
        """
        
        params = [query_embedding]
        conditions = []
        
        if category:
            conditions.append("category = %s")
            params.append(category)
            
        if price_range:
            conditions.append("price BETWEEN %s AND %s")
            params.extend(price_range)
        
        if conditions:
            base_sql += " AND " + " AND ".join(conditions)
            
        base_sql += """
        ORDER BY description_embedding <=> %s::vector
        LIMIT %s;
        """
        
        params.extend([query_embedding, limit])
        
        cur = self.conn.cursor()
        try:
            cur.execute(base_sql, params)
            results = cur.fetchall()
            cur.close()
            return results
        except Exception as e:
            print(f"❌ 混合搜索失败: {e}")
            cur.close()
            return []
    
    def get_database_stats(self):
        """获取数据库统计信息"""
        cur = self.conn.cursor()
        
        # 产品总数
        cur.execute("SELECT COUNT(*) FROM products;")
        product_count = cur.fetchone()[0]
        
        # 有嵌入向量的产品数
        cur.execute("SELECT COUNT(*) FROM products WHERE description_embedding IS NOT NULL;")
        embedded_count = cur.fetchone()[0]
        
        # 用户行为总数
        cur.execute("SELECT COUNT(*) FROM user_behaviors;")
        behavior_count = cur.fetchone()[0]
        
        # 索引信息
        cur.execute("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename = 'products' AND indexname LIKE '%embedding%';
        """)
        indexes = cur.fetchall()
        
        cur.close()
        
        return {
            "total_products": product_count,
            "embedded_products": embedded_count,
            "total_behaviors": behavior_count,
            "vector_indexes": indexes
        }
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("✅ 数据库连接已关闭")


def format_products(products: List[Tuple]) -> str:
    """格式化产品列表输出"""
    if not products:
        return "❌ 没有找到相关产品"
    
    result = "\n" + "="*60 + "\n"
    for i, product in enumerate(products, 1):
        if len(product) == 8:  # 包含相似度分数
            id, name, desc, category, price, brand, tags, similarity = product
            result += f"{i}. {name} (相似度: {similarity:.3f})\n"
        else:
            id, name, desc, category, price, brand, tags = product
            result += f"{i}. {name}\n"
            
        result += f"   分类: {category} | 品牌: {brand} | 价格: ¥{price}\n"
        result += f"   描述: {desc}\n"
        result += f"   标签: {tags}\n"
        result += "-" * 60 + "\n"
    
    return result


def main():
    """主函数演示完整功能"""
    
    # 数据库配置（请根据实际情况修改）
    db_config = {
        "host": "10.102.35.47",
        "database": "test", 
        "user": "opentenbase",
        "port": 30004,
        "client_encoding": "utf8"
    }
    
    # 初始化推荐系统
    print("🚀 初始化产品推荐系统...")
    recommender = ProductRecommendationSystem(db_config)
    
    try:
        # 1. 连接数据库
        recommender.connect_db()
        
        # 2. 设置数据库结构
        print("\n📦 设置数据库结构...")
        recommender.setup_database()
        
        # 3. 插入示例数据
        print("\n📝 插入示例产品数据...")
        recommender.insert_sample_products()
        
        print("\n👤 插入示例用户行为数据...")
        recommender.insert_sample_user_behaviors()
        
        # 4. 创建向量索引
        print("\n🔍 创建向量索引...")
        recommender.create_vector_index()
        
        # 5. 展示数据库统计
        print("\n📊 数据库统计信息:")
        stats = recommender.get_database_stats()
        print(f"   总产品数: {stats['total_products']}")
        print(f"   已嵌入产品数: {stats['embedded_products']}")
        print(f"   用户行为记录数: {stats['total_behaviors']}")
        print(f"   向量索引数: {len(stats['vector_indexes'])}")
        
        # 6. 演示语义搜索
        print("\n🔍 演示1: 语义搜索")
        print("搜索查询: '适合办公的轻薄笔记本电脑'")
        search_results = recommender.semantic_search("适合办公的轻薄笔记本电脑", limit=3)
        print(format_products(search_results))
        
        # 7. 演示个性化推荐
        print("\n🎯 演示2: 基于用户历史的个性化推荐")
        print("为用户1推荐产品(该用户喜欢iPhone和AirPods):")
        user_recommendations = recommender.recommend_by_user_history(user_id=1, limit=3)
        print(format_products(user_recommendations))
        
        # 8. 演示混合搜索
        print("\n🔀 演示3: 混合搜索（语义+筛选）")
        print("搜索: '拍照手机'，类别: '手机'，价格: 5000-8000元")
        hybrid_results = recommender.hybrid_search(
            query="拍照手机", 
            category="手机", 
            price_range=(5000, 8000),
            limit=3
        )
        print(format_products(hybrid_results))
        
        # 9. 展示向量距离计算
        print("\n📐 演示4: 向量距离分析")
        print("查询 '无线耳机' 与各产品的相似度:")
        wireless_results = recommender.semantic_search("无线耳机", limit=8)
        for product in wireless_results:
            name, similarity = product[1], product[7]
            print(f"   {name}: {similarity:.4f}")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        recommender.close_connection()


if __name__ == "__main__":
    """
    使用说明:
    
    1. 安装依赖:
       pip install psycopg2-binary sentence-transformers numpy
    
    2. 设置PostgreSQL:
       - 安装PostgreSQL数据库
       - 安装pgvector扩展: 
         CREATE EXTENSION vector;
    
    3. 修改数据库配置:
       更新main()函数中的db_config字典
    
    4. 运行程序:
       python pgvector_demo.py
    
    业务应用场景:
    - 电商产品推荐
    - 内容推荐系统  
    - 文档相似度搜索
    - 知识库问答系统
    - 图像搜索引擎
    """
    
    print("""
    🛍️  OpenTenbase + pgvector 产品推荐系统演示
    ===============================================
    
    本Demo展示了如何使用OpenTenBase + pgvector构建一个完整的产品推荐系统, 包括:
    ✨ 本地文本嵌入向量生成
    ✨ OpenTenBase向量数据库存储
    ✨ 高效向量索引
    ✨ 语义相似度搜索
    ✨ 个性化推荐
    ✨ 混合搜索（语义+传统筛选）
    """)
    
    main()