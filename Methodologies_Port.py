#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 10:38:45 2025

@author: camilocastillo
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Dict, Optional
import yahoo_data
import metrics_portfolio

class Do_portfolio:
    def __init__(self, tickers:List[str], start: str = '2024-01-01', end: Optional[str] = None):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.weights = None
        self.info = yahoo_data.yahoo_data(tickers=self.tickers,
                                          start=self.start,
                                          end = self.end)
    
    def random_weights(self,n_port:int, short: bool = True, seed: Optional[int] = None) -> np.ndarray:
        "Genera pesos aleatorios que suman 1"
        rng = np.random.default_rng(seed)
        if not short:
            W = rng.dirichlet(np.ones(len(self.tickers)), size = n_port)
        else:
            X = rng.normal(size = (n_port,len(self.tickers)))
            W = X/np.sum(np.abs(X), axis = 1, keepdims= True)
        return W
    
    def Monte_Carlo(self, n_port: int, short: bool = True, seed: Optional[int] = None, rf: float = 0.01, periods_per_year: int = 252):
        "Construye un portafolio por Monte Carlos con el Max. Sharpe"
        W = self.random_weights(len(self.weights),n_port,short = short,seed = seed)
        results = np.zeros((n_port,4))
        for i in range(n_port):
            m = metrics_portfolio.annualize_stats(self.info.compute_returns(), W[i], rf= rf, periods_per_year=periods_per_year)
            results[i,0] = m.ann_return
            results[i,1] = m.ann_vol
            results[i,2] = m.sharpe
            results[i,3] = m.cv
        
        df = pd.DataFrame(results, columns = ['ann_returns', 'ann_vol', 'Sharpe', 'CV'])
        
        for j, t in enumerate(self.tickers):
            df[f'w_{t}'] = W[:,j]
            
        