import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import os

# 设置随机种子以确保结果可重现
tf.random.set_seed(42)
np.random.seed(42)


class MLPAttentionModelWithFFT:
    def __init__(self, use_magnitude_phase=True):
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0.1, 1.5))
        self.use_magnitude_phase = use_magnitude_phase  # 控制使用幅度/相位还是实部/虚部
        self.original_input_dim = None
        self.fft_input_dim = None

    def load_data(self, s_data_path, n_data_path):
        """加载训练数据"""
        print("Loading data...")
        try:
            try:
                s_data = np.loadtxt(s_data_path, delimiter=',')
            except:
                s_data = np.genfromtxt(s_data_path, delimiter=',')
                if np.isnan(s_data).any() or np.isinf(s_data).any():
                    s_data = np.genfromtxt(s_data_path, delimiter=None)

            try:
                n_data = np.loadtxt(n_data_path, delimiter=',')
            except:
                n_data = np.genfromtxt(n_data_path, delimiter=',')
                if np.isnan(n_data).any() or np.isinf(n_data).any():
                    n_data = np.genfromtxt(n_data_path, delimiter=None)

            print(f"s_data shape: {s_data.shape}")
            print(f"n_data shape: {n_data.shape}")

            if len(s_data.shape) == 1:
                s_data = s_data.reshape(-1, 1)
            if len(n_data.shape) == 1:
                n_data = n_data.reshape(-1, 1)

            return s_data, n_data
        except Exception as e:
            print(f"Data loading error: {e}")
            return self.load_data_manual(s_data_path, n_data_path)

    def load_data_manual(self, s_data_path, n_data_path):
        """手动解析数据文件"""
        try:
            s_lines = []
            with open(s_data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            if ',' in line:
                                values = [float(x.strip()) for x in line.split(',')]
                            elif '\t' in line:
                                values = [float(x.strip()) for x in line.split('\t')]
                            else:
                                values = [float(x.strip()) for x in line.split()]
                            s_lines.append(values)
                        except:
                            continue
            s_data = np.array(s_lines)

            n_lines = []
            with open(n_data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            if ',' in line:
                                values = [float(x.strip()) for x in line.split(',')]
                            elif '\t' in line:
                                values = [float(x.strip()) for x in line.split('\t')]
                            else:
                                values = [float(x.strip()) for x in line.split()]
                            n_lines.append(values)
                        except:
                            continue
            n_data = np.array(n_lines)

            print(f"Manual parsing completed: s_data {s_data.shape}, n_data {n_data.shape}")
            return s_data, n_data
        except Exception as e:
            print(f"Manual parsing failed: {e}")
            raise

    def apply_fft_to_data(self, s_data):
        """
        对输入数据进行傅里叶变换。
        """
        print("Applying FFT to input data...")
        self.original_input_dim = s_data.shape[1]

        # 对每一行（每个样本）进行FFT
        # s_data shape: (n_samples, n_features)
        fft_result = np.fft.fft(s_data, axis=1)  # shape: (n_samples, n_features)

        if self.use_magnitude_phase:
            # 使用幅度和相位
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)
            # 将幅度和相位连接起来作为新特征
            # shape: (n_samples, 2 * n_features)
            fft_features = np.concatenate([magnitude, phase], axis=1)
            print(f"  Using Magnitude and Phase. New feature dim: {fft_features.shape[1]}")
        else:
            # 使用实部和虚部
            real_part = np.real(fft_result)
            imag_part = np.imag(fft_result)
            # 将实部和虚部连接起来作为新特征
            # shape: (n_samples, 2 * n_features)
            fft_features = np.concatenate([real_part, imag_part], axis=1)
            print(f"  Using Real and Imaginary parts. New feature dim: {fft_features.shape[1]}")

        self.fft_input_dim = fft_features.shape[1]
        return fft_features

    def create_model(self, input_dim, output_dim):
        """创建MLP + 注意力机制模型"""
        # 1. 输入层
        inputs = tf.keras.Input(shape=(input_dim,), name='fft_input')

        # 2. MLP 特征提取
        # 第一个MLP块
        x = tf.keras.layers.Dense(512, activation='relu', name='mlp_1')(inputs)
        x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)

        # 第二个MLP块
        x = tf.keras.layers.Dense(256, activation='relu', name='mlp_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout_2')(x)

        # 第三个MLP块 - 用于注意力
        x = tf.keras.layers.Dense(128, activation='relu', name='mlp_3')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
        x = tf.keras.layers.Dropout(0.2, name='dropout_3')(x)

        # 3. 简单的自注意力机制 (Self-Attention)
        # 为了简化，我们可以将当前的 MLP 输出看作一个序列（长度为1，维度为128）
        # 或者，我们可以引入一个“特征维度”的注意力。
        # 这里我们实现一个基于特征维度的注意力，类似于SENet中的通道注意力。

        # 将 x 重塑以便于计算注意力
        # x shape: (batch, 128)
        # 为了计算特征注意力，我们将其视为 (batch, 1, 128)
        x_reshaped = tf.keras.layers.Reshape((1, 128))(x)  # (batch, 1, 128)

        # 注意力机制部分 (简化版 Squeeze-and-Excitation)
        attention_input = tf.keras.layers.GlobalAveragePooling1D()(x_reshaped)  # (batch, 128)
        attention_hidden = tf.keras.layers.Dense(64, activation='relu', name='attention_hidden')(
            attention_input)  # (batch, 64)
        attention_weights = tf.keras.layers.Dense(128, activation='sigmoid', name='attention_weights')(
            attention_hidden)  # (batch, 128)

        # 应用注意力权重
        # x shape: (batch, 128)
        # attention_weights shape: (batch, 128)
        x_attention_applied = tf.keras.layers.Multiply()([x, attention_weights])  # (batch, 128)

        # 4. 全连接层
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x_attention_applied)
        x = tf.keras.layers.BatchNormalization(name='bn_dense_1')(x)
        x = tf.keras.layers.Dropout(0.2, name='dropout_4')(x)

        x = tf.keras.layers.Dense(64, activation='relu', name='dense_64')(x)
        x = tf.keras.layers.Dropout(0.2, name='dropout_5')(x)

        # 5. 输出层
        outputs_scaled = tf.keras.layers.Dense(output_dim, activation='sigmoid', name='output_scaled')(x)

        # 6. 创建模型
        model = tf.keras.Model(inputs=inputs, outputs=outputs_scaled, name='mlp_attention_fft_model')

        # 7. 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='mae',
            metrics=['mae']
        )

        return model

    def safe_preprocess_data(self, s_data, n_data):
        """安全的数据预处理"""
        print("Preprocessing data...")
        try:
            print(f"Original s_data range: [{np.min(s_data):.6f}, {np.max(s_data):.6f}]")
            print(f"Original n_data range: [{np.min(n_data):.6f}, {np.max(n_data):.6f}]")

            # 1. 对 s_data 进行 FFT 变换
            s_data_fft = self.apply_fft_to_data(s_data)

            # 2. 处理 FFT 后的输入数据 - 标准化
            s_data_fft_clean = np.copy(s_data_fft)
            s_data_fft_clean = np.nan_to_num(s_data_fft_clean, nan=0.0,
                                             posinf=np.nanmax(s_data_fft_clean[np.isfinite(s_data_fft_clean)]),
                                             neginf=np.nanmin(s_data_fft_clean[np.isfinite(s_data_fft_clean)]))
            s_data_fft_scaled = self.scaler_x.fit_transform(s_data_fft_clean)

            # 3. 处理输出数据 n_data - 确保在0.1-1.5范围内并进行标准化
            n_data_clean = np.copy(n_data)
            n_data_clean = np.nan_to_num(n_data_clean, nan=0.8, posinf=1.5, neginf=0.1)
            n_data_clean = np.clip(n_data_clean, 0.1, 1.5)
            n_data_scaled_for_training = (n_data_clean - 0.1) / (1.5 - 0.1)

            print(f"Processed FFT s_data range: [{np.min(s_data_fft_scaled):.6f}, {np.max(s_data_fft_scaled):.6f}]")
            print(
                f"Processed n_data training range (0-1): [{np.min(n_data_scaled_for_training):.6f}, {np.max(n_data_scaled_for_training):.6f}]")
            print(
                f"Processed n_data original range (0.1-1.5): [{np.min(n_data_clean):.6f}, {np.max(n_data_clean):.6f}]")

            return s_data_fft_scaled, n_data_scaled_for_training, n_data_clean
        except Exception as e:
            print(f"Data preprocessing error: {e}")
            raise

    def train_model(self, s_data, n_data, epochs=150, batch_size=32, validation_split=0.2):
        """训练模型"""
        print("Training model...")
        s_data_fft_scaled, n_data_scaled_for_training, n_data_original = self.safe_preprocess_data(s_data, n_data)
        self.n_data_original = n_data_original

        self.model = self.create_model(s_data_fft_scaled.shape[1], n_data_scaled_for_training.shape[1])
        print(self.model.summary())

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]

        start_time = time.time()
        try:
            history = self.model.fit(
                s_data_fft_scaled, n_data_scaled_for_training,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training completed, time: {training_time:.2f} seconds")
            return history, training_time, s_data_fft_scaled, n_data_scaled_for_training
        except Exception as e:
            print(f"Training error: {e}")
            raise

    def evaluate_model(self, s_data_fft_scaled, n_data_scaled_for_training):
        """评估模型性能"""
        print("Evaluating model...")
        try:
            predictions_scaled = self.model.predict(s_data_fft_scaled, verbose=0)

            predictions_original = predictions_scaled * (1.5 - 0.1) + 0.1
            true_original = n_data_scaled_for_training * (1.5 - 0.1) + 0.1

            predictions_original = np.clip(predictions_original, 0.1, 1.5)
            true_original = np.clip(true_original, 0.1, 1.5)

            mse = np.mean((predictions_original - true_original) ** 2)
            mae = np.mean(np.abs(predictions_original - true_original))
            correlation = np.corrcoef(predictions_original.flatten(), true_original.flatten())[0, 1]

            print(f"Mean Squared Error (MSE): {mse:.6f}")
            print(f"Mean Absolute Error (MAE): {mae:.6f}")
            print(f"Correlation Coefficient: {correlation:.4f}")
            return predictions_original, true_original, mse, mae, correlation
        except Exception as e:
            print(f"Model evaluation error: {e}")
            raise

    def plot_metrics(self, history, training_time, mse, mae, correlation):
        """绘制训练过程图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history: axes[0, 0].plot(history.history['val_loss'], label='Validating Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history: axes[0, 1].plot(history.history['val_mae'], label='Validating MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        if 'lr' in history.history:
            axes[0, 2].plot(history.history['lr'])
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True)
        else:
            axes[0, 2].text(0.5, 0.5, 'Learning rate info not available', ha='center', va='center')

        metrics_names = ['MSE', 'MAE', 'Correlation']
        metrics_values = [mse, mae, correlation]
        colors = ['red', 'blue', 'green']
        axes[1, 0].bar(metrics_names[0], metrics_values[0], color=colors[0])
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].bar(metrics_names[1], metrics_values[1], color=colors[1])
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 2].bar(metrics_names[2], metrics_values[2], color=colors[2])
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].set_ylim(-1, 1)
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'Model Training Results - Time: {training_time:.2f}s', fontsize=16)
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def load_t_data(self, t_data_path):
        """加载测试数据"""
        print("Loading test data...")
        try:
            try:
                t_data = np.loadtxt(t_data_path, delimiter=',')
            except:
                t_data = np.genfromtxt(t_data_path, delimiter=',')
                if np.isnan(t_data).any() or np.isinf(t_data).any():
                    t_data = np.genfromtxt(t_data_path, delimiter=None)

            if len(t_data.shape) == 1:
                t_data = t_data.reshape(-1, 1)
            print(f"t_data shape: {t_data.shape}")
            return t_data
        except Exception as e:
            print(f"t_data loading error: {e}")
            return self.load_t_data_manual(t_data_path)

    def load_t_data_manual(self, t_data_path):
        """手动解析t_data文件"""
        try:
            t_lines = []
            with open(t_data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            if ',' in line:
                                values = [float(x.strip()) for x in line.split(',')]
                            elif '\t' in line:
                                values = [float(x.strip()) for x in line.split('\t')]
                            else:
                                values = [float(x.strip()) for x in line.split()]
                            t_lines.append(values)
                        except:
                            continue
            t_data = np.array(t_lines)
            print(f"Manual parsing t_data completed: {t_data.shape}")
            return t_data
        except Exception as e:
            print(f"Manual parsing t_data failed: {e}")
            raise

    def predict_and_save(self, t_data_path, output_path):
        """对新数据进行预测并保存结果"""
        try:
            t_data = self.load_t_data(t_data_path)

            # 1. 对测试数据进行 FFT 变换
            t_data_fft = self.apply_fft_to_data(t_data)

            # 2. 处理 FFT 后的输入数据 - 标准化 (使用训练时的 scaler)
            t_data_fft_clean = np.copy(t_data_fft)
            t_data_fft_clean = np.nan_to_num(t_data_fft_clean, nan=0.0,
                                             posinf=np.nanmax(t_data_fft_clean[np.isfinite(t_data_fft_clean)]),
                                             neginf=np.nanmin(t_data_fft_clean[np.isfinite(t_data_fft_clean)]))
            t_data_fft_scaled = self.scaler_x.transform(t_data_fft_clean)

            print("Making predictions...")
            predictions_scaled = self.model.predict(t_data_fft_scaled, verbose=0)

            predictions_original = predictions_scaled * (1.5 - 0.1) + 0.1
            predictions_clipped = np.clip(predictions_original, 0.1, 1.5)

            np.savetxt(output_path, predictions_clipped, fmt='%.6f')
            print(f"Predictions saved to: {output_path}")

            print("Prediction results (range: 0.1-1.5):")
            for i, pred in enumerate(predictions_clipped):
                print(f"Sample {i + 1}: {pred}")
                print(f"  Range check: min={np.min(pred):.4f}, max={np.max(pred):.4f}")

            return predictions_clipped
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    # use_magnitude_phase=True 使用幅度和相位
    # use_magnitude_phase=False 使用实部和虚部
    model_instance = MLPAttentionModelWithFFT(use_magnitude_phase=True)
    try:
        s_data_path = 's_data.txt'
        n_data_path = 'n_data.txt'
        s_data, n_data = model_instance.load_data(s_data_path, n_data_path)

        history, training_time, s_data_fft_scaled, n_data_scaled = model_instance.train_model(s_data, n_data, epochs=50)

        predictions, true_values, mse, mae, correlation = model_instance.evaluate_model(s_data_fft_scaled,
                                                                                        n_data_scaled)
        model_instance.plot_metrics(history, training_time, mse, mae, correlation)

        model_instance.model.save('trained_mlp_attention_fft_model.h5')
        print("Model saved as: trained_mlp_attention_fft_model.h5")

        t_data_path = 't_data.txt'
        output_path = 'predictions.txt'
        predictions_result = model_instance.predict_and_save(t_data_path, output_path)
        print("Program execution completed!")
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


# 创建示例数据文件（如果需要）
def create_sample_data():
    """创建示例数据文件用于测试"""
    print("Creating sample data files...")
    # 创建示例s_data (2000*201) - 使用随机数模拟信号
    s_data = np.random.randn(2000, 201) * 0.1
    np.savetxt('s_data.txt', s_data, fmt='%.6e', delimiter=',')
    # 创建示例n_data (2000*5)
    n_data = np.random.rand(2000, 5) * 1.4 + 0.1
    np.savetxt('n_data.txt', n_data, fmt='%.6f', delimiter=',')
    # 创建示例t_data (6*201)
    t_data = np.random.randn(6, 201) * 0.1
    np.savetxt('t_data.txt', t_data, fmt='%.6e', delimiter=',')
    print("Sample data files created!")


if __name__ == "__main__":
    # create_sample_data() # Uncomment to create sample data
    main()