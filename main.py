import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, List, Tuple, Optional, Callable

# 设置随机种子以确保结果可复现
np.random.seed(42)


class ThreeLayerNet:
    """三层神经网络：输入层 -> 隐藏层 -> 输出层"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 activation: str = 'relu', weight_scale: float = 1e-3):
        """
        初始化神经网络参数

        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度（类别数）
            activation: 激活函数类型，支持'relu'或'sigmoid'
            weight_scale: 权重初始化缩放因子
        """
        self.params = {}
        self.params['W1'] = weight_scale * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_scale * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.activation = activation

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        前向传播计算

        参数:
            X: 输入数据，形状 (N, D)

        返回:
            scores: 输出分数，形状 (N, C)
            cache: 缓存中间变量用于反向传播
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层：线性变换 + 激活函数
        z1 = X.dot(W1) + b1
        if self.activation == 'relu':
            a1 = np.maximum(0, z1)
        elif self.activation == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")

        # 第二层：线性变换（输出层）
        scores = a1.dot(W2) + b2

        cache = {'X': X, 'z1': z1, 'a1': a1, 'W2': W2}
        return scores, cache

    def backward(self, dscores: np.ndarray, cache: Dict, reg: float = 0.0) -> Dict:
        """
        反向传播计算梯度

        参数:
            dscores: 输出分数的梯度，形状 (N, C)
            cache: 前向传播时缓存的中间变量
            reg: L2正则化强度

        返回:
            grads: 各参数的梯度字典
        """
        X, z1, a1, W2 = cache['X'], cache['z1'], cache['a1'], cache['W2']
        N = X.shape[0]

        # 第二层梯度
        dW2 = a1.T.dot(dscores) / N + reg * W2
        db2 = np.sum(dscores, axis=0) / N

        # 第一层梯度
        da1 = dscores.dot(W2.T)
        if self.activation == 'relu':
            dz1 = da1 * (z1 > 0)
        elif self.activation == 'sigmoid':
            sigmoid_z1 = 1 / (1 + np.exp(-z1))
            dz1 = da1 * sigmoid_z1 * (1 - sigmoid_z1)

        dW1 = X.T.dot(dz1) / N + reg * self.params['W1']
        db1 = np.sum(dz1, axis=0) / N

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return grads


def softmax_loss(scores: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算softmax损失和梯度

    参数:
        scores: 模型输出分数，形状 (N, C)
        y: 真实标签，形状 (N,)

    返回:
        loss: 损失值
        dscores: 分数的梯度
    """
    N = scores.shape[0]

    # 数值稳定的softmax
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis=1, keepdims=True)

    # 计算损失
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    # 计算梯度
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N

    return loss, dscores


def train(
        model: ThreeLayerNet, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95, reg: float = 1e-5, num_epochs: int = 10,
        batch_size: int = 200, verbose: bool = True, save_path: Optional[str] = None
) -> Dict:
    """
    训练神经网络

    参数:
        model: 神经网络模型
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
        learning_rate: 学习率
        learning_rate_decay: 学习率衰减因子
        reg: L2正则化强度
        num_epochs: 训练轮数
        batch_size: 批次大小
        verbose: 是否打印训练进度
        save_path: 保存最佳模型的路径

    返回:
        history: 训练历史记录
    """
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    num_iterations = num_epochs * iterations_per_epoch

    best_val_acc = 0.0
    best_params = {}

    # 记录训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        # 每个epoch打乱数据顺序
        indices = np.random.permutation(num_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for iter in range(iterations_per_epoch):
            # 获取当前批次数据
            start_idx = iter * batch_size
            end_idx = min((iter + 1) * batch_size, num_train)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            # 前向传播
            scores, cache = model.forward(X_batch)
            loss, dscores = softmax_loss(scores, y_batch)

            # 加上正则化损失
            reg_loss = 0.5 * reg * (
                    np.sum(model.params['W1'] ** 2) +
                    np.sum(model.params['W2'] ** 2)
            )
            total_loss = loss + reg_loss

            # 反向传播
            grads = model.backward(dscores, cache, reg)

            # 参数更新（SGD）
            for param_name in model.params:
                model.params[param_name] -= learning_rate * grads[param_name]

        # 每个epoch结束后评估模型
        train_acc = (predict(model, X_train[:1000]) == y_train[:1000]).mean()
        val_acc = (predict(model, X_val) == y_val).mean()

        # 计算验证集损失
        val_scores, _ = model.forward(X_val)
        val_loss, _ = softmax_loss(val_scores, y_val)
        val_loss += 0.5 * reg * (
                np.sum(model.params['W1'] ** 2) +
                np.sum(model.params['W2'] ** 2)
        )

        # 记录历史
        history['train_loss'].append(total_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc and save_path is not None:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}
            save_model(model, save_path)

        # 学习率衰减
        learning_rate *= learning_rate_decay

        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, loss: {total_loss:.4f}, '
                  f'train acc: {train_acc:.4f}, val acc: {val_acc:.4f}, '
                  f'learning rate: {learning_rate:.6f}')

    # 加载最佳模型参数
    if save_path is not None and best_params:
        model.params = best_params

    return history


def predict(model: ThreeLayerNet, X: np.ndarray) -> np.ndarray:
    """
    使用模型进行预测

    参数:
        model: 训练好的神经网络模型
        X: 输入数据

    返回:
        y_pred: 预测标签
    """
    scores, _ = model.forward(X)
    y_pred = np.argmax(scores, axis=1)
    return y_pred


def save_model(model: ThreeLayerNet, path: str) -> None:
    """保存模型参数到文件"""
    with open(path, 'wb') as f:
        pickle.dump(model.params, f)


def load_model(model: ThreeLayerNet, path: str) -> None:
    """从文件加载模型参数"""
    with open(path, 'rb') as f:
        model.params = pickle.load(f)


def get_cifar10_data(num_training: int = 49000, num_validation: int = 1000,
                     num_test: int = 1000, subtract_mean: bool = True) -> Dict:
    """
    加载CIFAR-10数据集并进行预处理

    参数:
        num_training: 训练样本数
        num_validation: 验证样本数
        num_test: 测试样本数
        subtract_mean: 是否减去均值

    返回:
        data: 包含预处理后数据集的字典
    """
    # 检查数据是否已下载
    if not os.path.exists('cifar-10-batches-py'):
        raise FileNotFoundError("CIFAR-10数据未找到，请先下载数据。")

    # 加载原始数据
    X_train, y_train, X_test, y_test = load_cifar10()

    # 划分数据集
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 预处理：展平图像
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 预处理：减去均值图像
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # 打包数据
    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }
    return data


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载CIFAR-10数据集"""
    X_train = []
    y_train = []

    # 加载训练批次
    for i in range(1, 6):
        filename = f'cifar-10-batches-py/data_batch_{i}'
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            y = datadict[b'labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            X_train.append(X)
            y_train.extend(y)

    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)

    # 加载测试批次
    filename = 'cifar-10-batches-py/test_batch'
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X_test = datadict[b'data']
        y_test = datadict[b'labels']
        X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def visualize_loss_accuracy(history: Dict) -> None:
    """可视化训练过程中的损失和准确率"""
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def visualize_weights(model: ThreeLayerNet) -> None:
    """可视化神经网络第一层权重"""
    W1 = model.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)

    plt.figure(figsize=(10, 10))
    for i in range(min(100, W1.shape[0])):
        plt.subplot(10, 10, i + 1)
        # 标准化权重以便可视化
        wmin, wmax = np.min(W1[i]), np.max(W1[i])
        img = 255.0 * (W1[i].squeeze() - wmin) / (wmax - wmin)
        plt.imshow(img.astype('uint8'))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('weight_visualization.png')
    plt.show()


def hyperparameter_search(data: Dict, hidden_sizes: List[int],
                          learning_rates: List[float], reg_strengths: List[float]) -> None:
    """
    超参数搜索

    参数:
        data: 数据集字典
        hidden_sizes: 隐藏层大小列表
        learning_rates: 学习率列表
        reg_strengths: 正则化强度列表
    """
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    results = {}
    best_val_acc = 0.0
    best_model = None

    for hs in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_strengths:
                print(f"训练模型: 隐藏层大小={hs}, 学习率={lr}, 正则化={reg}")

                # 初始化模型
                model = ThreeLayerNet(input_size=32 * 32 * 3, hidden_size=hs, output_size=10)

                # 训练模型
                history = train(
                    model, X_train, y_train, X_val, y_val,
                    learning_rate=lr, reg=reg, num_epochs=5, verbose=False
                )

                # 评估模型
                val_acc = (predict(model, X_val) == y_val).mean()
                print(f"验证准确率: {val_acc:.4f}")

                # 记录结果
                results[(hs, lr, reg)] = val_acc

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model

    # 打印结果
    print("\n超参数搜索结果:")
    for (hs, lr, reg), val_acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"隐藏层大小={hs}, 学习率={lr}, 正则化={reg}: 验证准确率={val_acc:.4f}")

    print(f"\n最佳验证准确率: {best_val_acc:.4f}")

    return best_model


def main():
    """主函数：运行完整的训练和评估流程"""
    # 加载数据
    print("加载CIFAR-10数据集...")
    data = get_cifar10_data()
    print(f"训练数据形状: {data['X_train'].shape}")
    print(f"验证数据形状: {data['X_val'].shape}")
    print(f"测试数据形状: {data['X_test'].shape}")

    # 超参数搜索
    print("\n开始超参数搜索...")
    hidden_sizes = [50, 100]
    learning_rates = [1e-3, 5e-3]
    reg_strengths = [1e-4, 1e-3]

    best_model = hyperparameter_search(
        data, hidden_sizes, learning_rates, reg_strengths
    )

    # 使用最佳超参数训练最终模型
    print("\n使用最佳超参数训练最终模型...")
    best_hidden_size = 100
    best_learning_rate = 5e-3
    best_reg = 1e-4

    final_model = ThreeLayerNet(
        input_size=32 * 32 * 3, hidden_size=best_hidden_size, output_size=10
    )

    history = train(
        final_model, data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        learning_rate=best_learning_rate,
        reg=best_reg,
        num_epochs=20,
        save_path='best_model.pkl'
    )

    # 可视化训练历史
    visualize_loss_accuracy(history)

    # 可视化权重
    visualize_weights(final_model)

    # 在测试集上评估模型
    test_acc = (predict(final_model, data['X_test']) == data['y_test']).mean()
    print(f"\n测试集准确率: {test_acc:.4f}")


if __name__ == "__main__":
    main()