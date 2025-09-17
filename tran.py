import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

# ================== 参数设置 ==================

class Args:
    evi_path = 'F:/transformer/火前/average_evi_mei32_sif.csv'
    prec_temp_path = 'F:/transformer/prec平均值/prec_mei32_prec.csv'
    window_size = 24
    train_years = 4
    predict_years = 4
    random_state = 34
    model_name = 'Transformer-ekan'
    dropout = 0.2
    hidden_dim = 32
    n_layers = 2
    num_epochs = 300
    seed = 1
    lr = 5e-4
    n_mc_samples = 100
    ci_alpha = 0.05
    smooth_method = 'moving_average'
    smooth_window = 5
    smooth_sigma = 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = Args()

# ========== 时间戳与结果保存路径 ==========

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base_save_dir = Path("F:/transformer/B913-KAN+Transformer时间序列预测完整/time_series/timeseries/examples/results/MultiFeatureTransformer")
save_dir = base_save_dir / f'{args.model_name}_{timestamp}'
save_dir.mkdir(parents=True, exist_ok=True)

# ================== 数据平滑函数 ==================

def apply_smoothing(data, method='moving_average', window_size=5, sigma=1.0):
    data_smoothed = data.copy()
    for col in data_smoothed.columns:
        if method == 'moving_average':
            data_smoothed[col] = data_smoothed[col].rolling(window=window_size, center=True, min_periods=1).mean()
        elif method == 'gaussian':
            data_smoothed[col] = gaussian_filter1d(data_smoothed[col].values, sigma=sigma, mode='reflect')
    return data_smoothed

# ================== 读取并预处理数据 ==================

def load_and_prepare(evi_path, prec_temp_path):
    # 读取EVI数据
    evi = pd.read_csv(evi_path)
    evi['Date'] = pd.to_datetime(evi['Date'], format='%Y%m')
    evi.set_index('Date', inplace=True)
    # 读取降水和温度数据
    pt = pd.read_csv(prec_temp_path)
    pt['Date'] = pd.to_datetime(pt['Date'], format='%Y%m')
    pt.set_index('Date', inplace=True)

    evi = apply_smoothing(evi, method=args.smooth_method, window_size=args.smooth_window, sigma=args.smooth_sigma)
    pt = apply_smoothing(pt,  method=args.smooth_method, window_size=args.smooth_window, sigma=args.smooth_sigma)

    df = pd.concat([evi[['evi']], pt[['prec', 'tem']]], axis=1)
    df['month']     = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    last_evi_date = evi.index[-1]
    return df, last_evi_date

# ================== 按年分组归一化并构建序列 ==================

def split_and_scale(df):
    months_in_year = 12
    train_months   = args.train_years * months_in_year
    predict_months = args.predict_years * months_in_year

    df_train   = df.iloc[:train_months]
    df_predict = df.iloc[train_months: train_months + predict_months]

    feat_scalers = {}
    tgt_scalers  = {}
    train_feat_scaled, train_tgt_scaled = [], []

    for year, grp in df_train.groupby(df_train.index.year):
        fsc = MinMaxScaler(feature_range=(-1,1))
        tsc = MinMaxScaler(feature_range=(-1,1))
        feats = grp[['prec','tem','month_sin','month_cos']].values
        tgts  = grp[['evi']].values
        train_feat_scaled.append(fsc.fit_transform(feats))
        train_tgt_scaled .append(tsc.fit_transform(tgts))
        feat_scalers[int(year)] = fsc
        tgt_scalers [int(year)] = tsc
    train_feat_scaled = np.vstack(train_feat_scaled)
    train_tgt_scaled  = np.vstack(train_tgt_scaled)

    predict_feat_scaled = []
    predict_years = []
    for year, grp in df_predict.groupby(df_predict.index.year):
        fsc = MinMaxScaler(feature_range=(-1,1))
        feats = grp[['prec','tem','month_sin','month_cos']].values
        predict_feat_scaled.append(fsc.fit_transform(feats))
        feat_scalers[int(year)] = fsc
        predict_years.extend([int(year)]*len(grp))
    predict_feat_scaled = np.vstack(predict_feat_scaled)
    predict_years = np.array(predict_years, dtype=int)

    X_train, Y_train = [], []
    for i in range(train_feat_scaled.shape[0] - args.window_size):
        X_train.append(train_feat_scaled[i:i+args.window_size])
        Y_train.append(train_tgt_scaled[i+args.window_size])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train, predict_feat_scaled, predict_years, feat_scalers, tgt_scalers

# ================== MC不确定性预测 ==================

def predict_future_with_uncertainty(model, last_window, future_feats, future_years, feat_scalers, tgt_scalers):
    model.eval()
    preds, lows, ups = [], [], []
    window = last_window.copy()

    last_train_year = max(tgt_scalers.keys())

    for i in range(future_feats.shape[0]):
        mc = []
        for _ in range(args.n_mc_samples):
            with torch.no_grad():
                inp = torch.from_numpy(window).float().unsqueeze(0).to(args.device)
                out = model(inp).cpu().numpy().flatten()
            mc.append(out)
        mc = np.stack(mc, axis=0)
        mean = mc.mean(axis=0)
        low  = np.percentile(mc, 100*args.ci_alpha/2, axis=0)
        high = np.percentile(mc, 100*(1-args.ci_alpha/2), axis=0)
        preds.append(mean); lows.append(low); ups.append(high)

        next_feat = future_feats[i].reshape(1,-1)
        window = np.vstack([window[1:], next_feat])

    preds_inv, lows_inv, ups_inv = [], [], []
    for i, yr in enumerate(future_years):
        scaler = tgt_scalers.get(int(yr), tgt_scalers[last_train_year])
        preds_inv.append(scaler.inverse_transform(preds[i].reshape(1,-1))[0])
        lows_inv.append(scaler.inverse_transform(lows[i].reshape(1,-1))[0])
        ups_inv .append(scaler.inverse_transform(ups[i].reshape(1,-1))[0])

    return np.array(preds_inv), np.array(lows_inv), np.array(ups_inv)

# PositionalEncoding 类定义
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# TimeSeriesTransformer_ekan 模型类定义
class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads=2, num_layers=1, num_outputs=1, hidden_space=32, dropout_rate=0.3):
        super(TimeSeriesTransformer_ekan, self).__init__()

        self.input_dim = input_dim
        self.model_dim = hidden_space

        self.input_projection = nn.Linear(input_dim, hidden_space)
        self.pos_encoder = PositionalEncoding(hidden_space, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,
            nhead=num_heads,
            dim_feedforward=hidden_space * 2,
            dropout=dropout_rate,
            activation='gelu',
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_space)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_space, hidden_space // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_space // 2, num_outputs)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)

        x = x[-1, :, :]
        x = self.norm(x)

        return self.decoder(x)

# ================== 主流程 ==================

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df, last_evi_date = load_and_prepare(args.evi_path, args.prec_temp_path)
    X_train_orig, Y_train_orig, X_pred_feats, pred_years, feat_scalers, tgt_scalers = split_and_scale(df)

    def augment_data(x, y, method='jitter', n_aug=3, noise_std=0.01):
        xs, ys = [x], [y]
        if method=='jitter':
            for _ in range(n_aug): xs.append(x + np.random.normal(0, noise_std, x.shape)); ys.append(y)
        elif method=='slice':
            max_shift = 3
            for s in range(1, max_shift+1):
                if s + args.window_size < x.shape[1]: xs.append(x[:, s:, :]); ys.append(y[s:])
        return np.vstack(xs), np.vstack(ys)

    X_train, Y_train = augment_data(X_train_orig, Y_train_orig, method='jitter', n_aug=3, noise_std=0.03)
    X_train, Y_train = augment_data(X_train,         Y_train,         method='slice')

    X_t = torch.from_numpy(X_train).float().to(args.device)
    Y_t = torch.from_numpy(Y_train).float().to(args.device)

    model = TimeSeriesTransformer_ekan(
        input_dim=4,
        num_heads=4,
        num_layers=args.n_layers,
        num_outputs=1,
        hidden_space=args.hidden_dim,
        dropout_rate=args.dropout
    ).to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_losses = []
    for ep in range(1, args.num_epochs+1):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, Y_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if ep % 50 == 0: print(f"Epoch {ep}/{args.num_epochs}, Loss={loss.item():.6f}")

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title('Training Loss')
    plt.savefig(save_dir / 'training_loss_curve.png')
    plt.close()

    model.eval()
    with torch.no_grad(): tr_pred_orig = model(torch.from_numpy(X_train_orig).float().to(args.device)).cpu().numpy()

    label_dates = df.index[:args.train_years*12][args.window_size:]
    tr_pred_inv, tr_true_inv = [], []
    for i, date in enumerate(label_dates):
        year = date.year
        scaler = tgt_scalers[int(year)]
        tr_pred_inv.append(scaler.inverse_transform(tr_pred_orig[i].reshape(1,-1))[0])
        tr_true_inv.append(scaler.inverse_transform(Y_train_orig[i].reshape(1,-1))[0])
    tr_pred_inv = np.vstack(tr_pred_inv); tr_true_inv = np.vstack(tr_true_inv)

    train_mae  = mean_absolute_error(tr_true_inv, tr_pred_inv)
    train_mse  = mean_squared_error(tr_true_inv, tr_pred_inv)
    train_rmse = np.sqrt(train_mse)
    train_r2   = r2_score(tr_true_inv, tr_pred_inv)

    pd.DataFrame({'MAE':[train_mae],'RMSE':[train_rmse],'R2':[train_r2]}).to_csv(save_dir / 'train_metrics.csv', index=False)
    plt.figure(); plt.plot(tr_true_inv, label='True'); plt.plot(tr_pred_inv, label='Predicted'); plt.legend(); plt.title('Train True vs Pred')
    plt.savefig(save_dir / 'train_true_vs_predicted.png'); plt.close()
    print(f"训练集 MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

    last_win = X_train[-1]
    preds, lower, upper = predict_future_with_uncertainty(
        model, last_win, X_pred_feats, pred_years, feat_scalers, tgt_scalers
    )
    future_dates = pd.date_range(
        start=last_evi_date + pd.offsets.MonthBegin(),
        periods=preds.shape[0],
        freq='ME'
    )
    future_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y%m'),
        'Predicted_EVI': preds.flatten(),
        'Lower_Bound': lower.flatten(),
        'Upper_Bound': upper.flatten()
    })
    future_df.to_csv(save_dir / 'future_predictions_with_uncertainty.csv', index=False)

    print("所有结果已保存至：", save_dir)

    if torch.cuda.is_available(): torch.cuda.empty_cache()