import numpy as np
import pandas as pd
import numpy.ma as ma
from numpy.linalg import inv, eig
from scipy.optimize import minimize, Bounds
from scipy.stats import skew, kurtosis

class StratsAnalyzer:
    def __init__(self):
        self._strats_lst = ["EW","MinVar","MV-S","RR-MaxRet","MinCVar","ERC","MD"]
    
    @staticmethod
    def average_returns(P):
        ret = P.pct_change()
        lret = np.log(ret)
        lret_mean = lret.mean()*12
        return np.exp(lret_mean) - 1
    
    @staticmethod
    def max_drawdown(V):
        return (V/V.cummax()-1).min()

    # objective functions for all strategies
    @staticmethod
    def MinVar_obj(weight, hist_ret):
        """
        minimize variance
        """
        return weight@hist_ret.cov()@weight.T
    
    @staticmethod
    def sum_to_1(weight):
        return abs(np.sum(weight) - 1) - 1e-15
    
    @staticmethod
    def MVS_obj(weight, hist_ret, adjusted=True):
        """
        maximize Sharpe ratio or adjusted Sharpe ratio
        """
        sr = (weight@hist_ret.mean())/np.sqrt(weight@hist_ret.cov()@weight.T)
        sample_skewness = np.mean(((hist_ret@weight.T - weight@hist_ret.mean())/np.sqrt(weight@hist_ret.cov()@weight.T))**3)
        sample_kurtosis = np.mean(((hist_ret@weight.T - weight@hist_ret.mean())/np.sqrt(weight@hist_ret.cov()@weight.T))**4)
        if adjusted:
            sr = sr*(1+sample_skewness/6*sr - sample_kurtosis/24*(sr**2))
        return -sr
    
    @staticmethod 
    def MaxRet_obj(weight, hist_ret):
        """
        maximize nominal returns
        """
        return -(weight@hist_ret.mean())
    
    @staticmethod 
    def MinCVar_obj(weight, hist_ret, alpha=0.05, gamma=1):
        """
        minimize Mean-CVar
        """
        rets = hist_ret@weight
        var = np.percentile(rets,alpha*100)
        exp_shortfall = -rets[rets<=var].mean()
        return -(rets.mean() - gamma/2*exp_shortfall)
    
    @staticmethod 
    def ERC_obj(weight, hist_ret):
        """
        minimize mean absolute error between risk contribution vectors and target equally weighted (risk contribution) vectors
        """
        n = hist_ret.shape[1]
        portfolio_vol = np.sqrt(weight@hist_ret.cov()@weight.T)
        risk_contrib = np.multiply(weight, (hist_ret.cov()@weight.T)/portfolio_vol)
        return np.mean(abs(risk_contrib-portfolio_vol/n))
    
    @staticmethod 
    def MD_obj(weight, hist_ret):
        """
        maximize the portfolio diversification index
        """
        n = hist_ret.shape[1]
        w, v = eig(np.multiply(weight, hist_ret).cov())
        normalizd_eig_vals = sorted(w,reverse=True)/np.sum(w)
        PDI = 2 * np.arange(1,n+1) @ normalizd_eig_vals.T - 1
        return -PDI
    

    # Functions to compute weights for each strategy
    # The functions below can be refactored/cleaned further easily for sure. For the purpose of clarity and considering time limitation, I decided to code them as its current style.
    def compute_MinVar_weight(self, hist_ret, is_constrainted=False, constraints=None, bounds=None):
        """
        Compute portfolio weights in MinVar portfolio
        Note that sample covariance is singular when number of dates < number of asset classes, a shrinkage method developed by Ledoit & Wolf (2004) can be used to mitigate this problem.
        By having a limited number of assets here, we do not have such problem.
        @params:
            hist_ret: ndarray, historical daily (log) returns in the shape of [# of dates, # of assets]
        """
        n = hist_ret.shape[1]
        weights = (inv(hist_ret.cov())@np.ones(n).T) / \
                    (np.ones(n)@inv(hist_ret.cov())@np.ones(n).T)
        if is_constrainted:
            assert constraints is not None, "please specify constraints"
            ans = minimize(self.MinVar_obj, x0=weights, args=(hist_ret), method = "SLSQP", bounds = bounds, constraints = constraints, tol=1e-8, options={"maxiter":1e5})
            if not ans.success: 
                print("Failure in Finding a numeric solution for MinVar")
                nonneg_weights = ma.filled(ma.masked_less_equal(weights, 0),0)
                weights = nonneg_weights/nonneg_weights.sum()
            else:
                weights = ans.x
        return weights

    def compute_MinCVar_weight(self, hist_ret, constraints=None, bounds=None, is_constrainted=True):
        """
        a strategy accounts for higher moments via Conditional VaR
        """
        n = hist_ret.shape[1]
        weights = self.compute_MVS_weight(hist_ret, is_constrainted=True, bounds = Bounds(0, np.inf), constraints = [{'type':'eq', 'fun': self.sum_to_1}])
        ans = minimize(self.MinCVar_obj, x0=weights, args=(hist_ret), method = "SLSQP", bounds = bounds, constraints = constraints, tol=1e-8, options={"maxiter":1e5})
        if not ans.success: 
            print("Failure in Finding a numeric solution for MinCVar")
            #print(ans)
            nonneg_weights = ma.filled(ma.masked_less_equal(weights, 0),0)
            weights = nonneg_weights/nonneg_weights.sum()
        else:
            weights = ans.x
        return weights

    def compute_ERC_weight(self, hist_ret,constraints=None, bounds=None, is_constrainted=True):
        """
        a strategy that allocates each asset same contribution to portfolio risk
        """
        n = hist_ret.shape[1]
        weights = np.ones(n)/n
        ans = minimize(self.ERC_obj, x0=weights, args=(hist_ret), method = "SLSQP", bounds = bounds, constraints = constraints, tol=1e-8, options={"maxiter":1e5})
        if ans.success:
            weights = ans.x
        return weights

    def compute_MD_weight(self, hist_ret,constraints=None, bounds=None, is_constrainted=True):
        """
        a strategy that maximize the Portfolio Diversification Index (PDI)
        """
        n = hist_ret.shape[1]
        weights = np.ones(n)/n
        ans = minimize(self.MD_obj, x0=weights, args=(hist_ret), method = "SLSQP", bounds = bounds, constraints = constraints, tol=1e-8, options={"maxiter":1e5})
        #_ans = None
        if ans.success:
            weights = ans.x
        return weights

    def compute_RRMaxRet_weight(self, hist_ret, constraints=None, bounds=None, is_constrainted=True):
        """
        a strategy that maximize the Portfolio returns
        """
        n = hist_ret.shape[1]
        weights = np.ones(n)/n
        #_ans = None
        ans = minimize(self.MaxRet_obj, x0= weights, args=(hist_ret), method = "SLSQP", bounds = bounds, constraints = constraints, tol=1e-8, options={"maxiter":1e5})
        if not ans.success: 
            print("Failure in Finding a numeric solution for RRMaxRet")
            nonneg_weights = ma.filled(ma.masked_less_equal(weights, 0),0)
            weights = nonneg_weights/nonneg_weights.sum()
        else:
            weights = ans.x
        return weights

    def compute_MVS_weight(self, hist_ret, is_constrainted=False, constraints=None, bounds=None):
        """
        a strategy that maximize the Portfolio risk-adjusted return
        """
        n = hist_ret.shape[1]
        target_ret = hist_ret.mean()
        weights = (inv(hist_ret.cov())@target_ret.T) / \
                    (np.ones(n)@inv(hist_ret.cov())@target_ret.T)
        if is_constrainted:
            assert constraints is not None, "please specify constraints"
            ans = minimize(self.MVS_obj, x0=weights, args=(hist_ret), method = "SLSQP", bounds = bounds, constraints = constraints, tol=1e-8, options={"maxiter":1e5})
            if not ans.success: 
                print("Failure in Finding a numeric solution for MVS")
                nonneg_weights = ma.filled(ma.masked_less_equal(weights, 0),0)
                weights = nonneg_weights/nonneg_weights.sum()
            else:
                weights = ans.x
        return weights
    
    @staticmethod
    def get_bootstrap_weights(hist_ret, single_model_weights, gamma=1, M=100, rebalance_days=30, lookback_days=360):
        print("Working on Combination of Model Weights through Bootstrapping...")
        n_model = len(single_model_weights)
        T, n = hist_ret.shape
        bootstrap_model_weights = np.zeros((T-lookback_days, n_model))
        for idx, i in enumerate(range(lookback_days, T, rebalance_days)):
            rolling_window = hist_ret.iloc[i-lookback_days:i,:]
            bootstrap_res_at_t = np.zeros(n_model)
            for i in range(M):
                sample_ret = hist_ret.iloc[np.random.choice(np.arange(0,lookback_days), rebalance_days),:].mean()
                loss_func_vals = [(sample_ret@val_dict["weight"][idx*rebalance_days] - \
                                    gamma/2*(val_dict["weight"][idx*rebalance_days]@rolling_window.cov()@val_dict["weight"][idx*rebalance_days].T))
                                    for strat, val_dict in single_model_weights.items()]
                unit_res = np.zeros(n_model)
                unit_res[np.argmax(loss_func_vals)] = 1
                bootstrap_res_at_t += unit_res
            bootstrap_res_at_t = bootstrap_res_at_t/M
            bootstrap_model_weights[idx*rebalance_days:min((idx+1)*rebalance_days, T),:] = bootstrap_res_at_t
        print("Bootstrapping Done!")
        return bootstrap_model_weights

    # driver functions to consolidate weights and rets
    def get_port_ret(self, log_ret_data, strats, rebalance_days=30, lookback_days=360, add_model_combination=True, single_model_weights=None):
        """
        @params:
            strats_lst: container for strategy labels, a string or a list of strings
        """
        _strat_port_weight_func_mapping = {
            "MinVar": self.compute_MinVar_weight,
            "MinCVar": self.compute_MinCVar_weight,
            "ERC": self.compute_ERC_weight,
            "MD": self.compute_MD_weight,
            "RR-MaxRet": self.compute_RRMaxRet_weight,
            "MV-S": self.compute_MVS_weight
        }

        res = {}
        daily_ret = None
        T, n = log_ret_data.shape
        if isinstance(strats, str):
            if strats == "EW":
                daily_ret = np.sum(np.exp(log_ret_data.iloc[lookback_days:,])*(1/n), axis=1)
                port_weights = np.ones((T-lookback_days, n))/n
            
            else:
                port_weights = np.zeros((T-lookback_days, n))
                for idx, i in enumerate(range(lookback_days, T)):
                    if idx % rebalance_days == 0:
                        #print(idx,i)
                        weights = _strat_port_weight_func_mapping[strats](log_ret_data.iloc[i-lookback_days:i,:],
                                                                        is_constrainted=True,
                                                                        constraints = [{'type':'eq', 'fun': self.sum_to_1}],
                                                                        bounds=Bounds(0, np.inf)
                        )
                        # hot fix for complex warning
                        if strats == "MD" and (sum(weights<1e-6)>n//2):
                            weights = port_weights[idx-1]
                    port_weights[idx] = weights
                daily_ret = np.sum(np.multiply(np.exp(log_ret_data.iloc[lookback_days:,]), port_weights),axis=1)
            
            res["weight"] = port_weights
            res["daily_ret"] = daily_ret
            return res
        else:
            for strat in strats:
                print(f"Working on {strat}...")
                res[strat] = self.get_port_ret(log_ret_data, strat, rebalance_days=rebalance_days, lookback_days=lookback_days)
            print("Done!")
            if single_model_weights is not None and isinstance(single_model_weights, dict):
                res.update(single_model_weights)
            # add combination of models
            if add_model_combination:
                print("Working on Model Combinations...")
                n_model = len(res)
                naive_combo_weights, bootstrap_combo_weights = np.zeros((T-lookback_days, n)), np.zeros((T-lookback_days, n))
                naive_combo_rets, bootstrap_combo_rets = np.zeros(T-lookback_days), np.zeros(T-lookback_days)
                naive_model_weights = np.ones(n_model)/n_model
                bootstrap_model_weights = self.get_bootstrap_weights(log_ret_data, res, gamma=1, M=100, rebalance_days=rebalance_days, lookback_days=lookback_days)
                
                for i, (strat, val_dict) in enumerate(res.items()):
                    naive_combo_weights += naive_model_weights[i]*val_dict["weight"]
                    naive_combo_rets += naive_model_weights[i]*val_dict["daily_ret"]
                    bootstrap_combo_weights += np.multiply(np.expand_dims(bootstrap_model_weights[:,i],axis=1), val_dict["weight"])
                    bootstrap_combo_rets += np.multiply(bootstrap_model_weights[:,i], val_dict["daily_ret"])
                
                res.update({"naive_combo": {}, "bootstrap_combo":{}})
                res["naive_combo"]["weight"] = naive_combo_weights
                res["naive_combo"]["daily_ret"] = naive_combo_rets
                res["bootstrap_combo"]["weight"] = bootstrap_combo_weights
                res["bootstrap_combo"]["daily_ret"] = bootstrap_combo_rets
            
            # convert to Dataframe:
            for strat, val_dict in res.items():
                val_dict["weight"] = pd.DataFrame(val_dict["weight"],index=log_ret_data.index[lookback_days:], columns = log_ret_data.columns)
                val_dict["daily_ret"] = pd.Series(val_dict["daily_ret"], index=log_ret_data.index[lookback_days:], name=strat)
            return res
    
    @staticmethod
    def rolling_sharpe(x):
        return np.sqrt(30) * x.mean() / x.std()
    
    @staticmethod
    def rolling_adj_sharpe(x):
        sr = x.mean() / x.std()
        return np.sqrt(30) *  sr * (1+ skew(x)/6 * sr - kurtosis(x)/24 * sr**2)

    @staticmethod
    def compute_rol_effN(row):
        return 1/np.sum(np.square(row))
    
    @staticmethod
    def compute_tto(strat_weights_df):
        """
        compute target turnover
        """
        res = 0
        df = strat_weights_df.drop_duplicates()
        n = len(df)
        if n > 1:
            for i in range(1,n):
                res+=np.sum(np.abs(df.iloc[i,:] - df.iloc[i-1,:]))
            res=res/n
        return res

    def compute_rol_PDI(self, weight, hist_ret, window_size):
        idx_loc = np.where(hist_ret.index == weight.name)[0][0]
        pdi = -self.MD_obj(weight.to_numpy(), hist_ret.iloc[(idx_loc - window_size):(idx_loc),:])
        return pdi