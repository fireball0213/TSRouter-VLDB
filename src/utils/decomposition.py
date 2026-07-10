import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import time
from statsmodels.tsa.seasonal import STL
import os
from utils.tools import dotdict
class SeriesDecomposition(nn.Module):
    def __init__(self, args):
        super(SeriesDecomposition, self).__init__()
        self.resid=args.resid
        self.args=args
        self.kernel_size = args.kernel_size
        self.times=args.trend_dec_times       
        self.seasonal_period = args.period
        self.decomp_method = args.decomp_method
        self.device = self._acquire_device()
        self.kernel_sizes = [(self.kernel_size - 1) * (2 ** i) + 1 for i in range(self.times)]
        self.first_execution = 0
        self.batch_size=args.batch_size
        self.root_path=args.root_path
        # self.dataset_name=args.data
        # self.save_path= f"{self.dataset_name}_{self.decomp_method}_ks{self.kernel_size}_sp{self.seasonal_period}_bs{self.batch_size}_sl{self.args.seq_len}_pl48"
        self.save_path= f"{self.decomp_method}_ks{self.kernel_size}_sp{self.seasonal_period}"
                                          

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    def forward(self, x):

        start_time = time.time()
        x = self._check_input(x)
        trend, seasonal, residual = self._decompose(x)
        x = self._check_input(x)
        trend=self._check_input(trend).to(self.device, non_blocking=True)
        seasonal=self._check_input(seasonal).to(self.device, non_blocking=True)
        residual=self._check_input(residual).to(self.device, non_blocking=True)
        if time.time() - start_time > 0.05:
            # print(f"Decom: { time.time() - start_time:.4f}s")
            pass
        # if self.first_execution<10:
        #     self._plot_results(x, trend, seasonal, residual,0,767)
        #     self.first_execution+=1
            # self.first_execution = False
        trend, seasonal, residual = self._handle_residual(trend, seasonal, residual, self.resid)
        trend=self._check_input(trend).to(self.device, non_blocking=True)
        seasonal=self._check_input(seasonal).to(self.device, non_blocking=True)
        residual=self._check_input(residual).to(self.device, non_blocking=True)
        return trend, seasonal,residual

    def _check_input(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x

    def _decompose(self, x):
        raise NotImplementedError

    def _handle_residual(self, trend, seasonal, residual, resid):
        if resid == 'trend':
            trend += residual
        elif resid == 'seasonal':
            seasonal += residual
        else:
            pass
        return trend, seasonal, residual

    def save_decomposition_results(self, trend, seasonal, residual, batch_id):
                                    
        results_dir = os.path.join(self.root_path, self.save_path)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        os.makedirs(results_dir, exist_ok=True)

                                                            
        file_prefix = f"batch_{batch_id}_"
        torch.save(trend, os.path.join(results_dir, file_prefix + 'trend.pt'))
        torch.save(seasonal, os.path.join(results_dir, file_prefix + 'seasonal.pt'))
        torch.save(residual, os.path.join(results_dir, file_prefix + 'residual.pt'))

    def load_decomposition_results(self, batch_id):
                                       
        results_dir = os.path.join(self.root_path, self.save_path)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        file_prefix = f"batch_{batch_id}_"

                                     
        try:
            trend = torch.load(os.path.join(results_dir, file_prefix + 'trend.pt'))
            seasonal = torch.load(os.path.join(results_dir, file_prefix + 'seasonal.pt'))
            residual = torch.load(os.path.join(results_dir, file_prefix + 'residual.pt'))
        except FileNotFoundError:
            print(f"Decomposition results for batch {batch_id} not found.")
            return None, None, None

        return trend, seasonal, residual

    def _plot_results(self, x, trend, seasonal, residual,t1,t2):
        plt.figure(figsize=(16, 5))

                                          
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            trend = trend.cpu().numpy()
            season = seasonal.cpu().numpy()
            resid = residual.cpu().numpy()
                          
        plot_column = -1
                       
        if len(x.shape) == 1:
                          
            if t2 > x.shape[-1]:
                t2 = x.shape[-1]
            x = x.reshape(1, -1,1)
            trend = trend.reshape(1, -1,1)
            season = seasonal.reshape(1, -1,1)
            resid = residual.reshape(1, -1,1)
        plt.plot(trend[0,t1:t2, plot_column], label='trend', color='red')
        plt.plot(season[0,t1:t2, plot_column], label='season', color='blue')
        plt.plot(resid[0,t1:t2, plot_column], label='resid', color='lightgreen')
        plt.plot(x[0,t1:t2, plot_column], label='original', color='grey')
        plt.title(self.decomp_method+': kernel_size='+str(int(self.kernel_size))+' period='+str(int(self.seasonal_period))+' times='+str(int(self.times)))
        plt.legend()
        plt.tight_layout()
             
        file_name = f"decomposition_{self.first_execution}.png"
        save_path = os.path.join(self.root_path, self.save_path)
        save_path=os.path.join(save_path,'plot_figs')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_dir = os.path.join(save_path, file_name)
        plt.savefig(save_dir)
        # plt.draw()
        # plt.show()


class MovingAvgDecomposition1(SeriesDecomposition):
    def __init__(self, args):
        super(MovingAvgDecomposition1, self).__init__(args)



    def _decompose(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 1:
            x = x.reshape(1, -1,1)
        trend = torch.zeros_like(x)
        x_residual = x.clone()
        for kernel_size in self.kernel_sizes:
            front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
            x_pad = torch.cat([front, x_residual, end], dim=1)                      
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
            current_trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
            trend += current_trend
            x_residual -=current_trend


        seasonal = x - trend
        residual = x - trend - seasonal
        return trend, seasonal, residual


class MovingAvgDecomposition2(SeriesDecomposition):
    def __init__(self,args):
        super(MovingAvgDecomposition2, self).__init__(args)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def _decompose(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 1:
            x = x.reshape(1, -1, 1)
        trend = torch.zeros_like(x)
        x_residual = x.clone()
        # padding = self.seasonal_period // 2
        # x_padded = F.pad(x, (0, 0, padding - 1, padding), mode='reflect')
        #
        # # Moving average using unfold
        # unfolded = x_padded.unfold(1, self.seasonal_period, 1)
        # moving_avg = unfolded.mean(dim=-1)
        # trend = moving_avg[:, :seq_len, :]
        for kernel_size in self.kernel_sizes:
            front = x_residual[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
            end = x_residual[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
            x_pad = torch.cat([front, x_residual, end], dim=1)
            current_trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
            trend += current_trend
            x_residual -= current_trend

        deseasonalized = x - trend

        # Seasonal
        seasonal = torch.zeros_like(x)
        batch_size, seq_len, channels = x.shape
        for i in range(channels):
            deseasonalized_channel = deseasonalized[:, :, i]
            seasonal_channel = seasonal[:, :, i]
            for k in range(self.seasonal_period):
                seasonal_indices = torch.arange(k, seq_len, self.seasonal_period)
                if len(seasonal_indices) > 0:
                    seasonal_mean = deseasonalized_channel[:, seasonal_indices].mean(dim=1, keepdim=True)
                    seasonal_channel[:, seasonal_indices] = seasonal_mean
            seasonal[:, :, i] = seasonal_channel

        residual = deseasonalized - seasonal

        return trend, seasonal, residual

class DFTSeriesDecomposition(SeriesDecomposition):
    def __init__(self, args):
        super(DFTSeriesDecomposition, self).__init__(args)
        self.top_k = args.top_k

    def _decompose(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        seasonal = torch.fft.irfft(xf)
        trend = x - seasonal
        residual = torch.zeros_like(x)
        return trend, seasonal, residual

class STLDecomposition(SeriesDecomposition):
    def __init__(self, args):
        super(STLDecomposition, self).__init__(args)


    def _decompose(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 1:
            x = x.reshape(1, -1, 1)
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        trend = np.zeros_like(x)
        seasonal = np.zeros_like(x)
        residual = np.zeros_like(x)
        # result = STL(x, period=self.seasonal_period).fit()
        batch_size, seq_len, channels = x.shape[0], x.shape[1], x.shape[2]
        for j in range(batch_size):
            for i in range(channels):
                result = STL(x[j, :, i], period=self.seasonal_period).fit()
                trend[j, :, i] = result.trend
                seasonal[j, :, i] = result.seasonal
                residual[j, :, i] = result.resid

        return trend, seasonal, residual

def decomposition_method(method_name, args):
    if method_name == 'MA1':
        return MovingAvgDecomposition1(args)
    elif method_name == 'MA2':
        return MovingAvgDecomposition2(args)
    elif method_name == 'DFT':
        return DFTSeriesDecomposition(args)
    elif method_name == 'STL':
        return STLDecomposition(args)
    else:
        raise ValueError(f"Unknown decomposition method: {method_name}")

      
if __name__ == '__main__':
    x = torch.randn(100000)          
    method_name = 'MA1'                                 
    for method_name in ['MA1', 'MA2', 'DFT', 'STL']:
        print(f"Decomposition method: {method_name}")
        args = dotdict({'kernel_size': 7, 'period': 24, 'top_k': 5, 'decomp_method': method_name,'trend_dec_times':1})
        decomp = decomposition_method(method_name, args)
        trend, seasonal, residual = decomp(x)

