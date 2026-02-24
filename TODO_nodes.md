 what i definitively need is some regime-change or "market character" indicator. The biggest edge one can have is      
  to bias the trading direction on the market direction ("the trend is your friend") So a breakout from  a long        
  range or a base is a very strong indicator - but relying on moving averages is too laggy. I would need another NN     
   that is trained to recognize the "character" of a stock - range-bound, trending, choppy, spiking and                 
  mean-returning, etc.. Is there something out there ? I was reading some  research papers in this direction, but       
  found no direct testable software projects                                                                            
                                                                                                                        
● You're identifying one of the hardest open problems in quantitative finance, and you're right that the open-source    
  landscape is surprisingly sparse given the research volume. Here's an honest map:
                                                                                                                        
  ---            
  What actually exists and is testable                                                                                  
                                                                                                                        
  1. Hidden Markov Models — hmmlearn

  The most battle-tested approach. You give it a feature vector (returns, volatility, volume) and it unsupervisedly
  learns N hidden states (e.g. bull/bear/chop). Simple, fast, well-understood failure modes.
  from hmmlearn import hmm
  # features: [log_return, rolling_vol, rolling_autocorr]
  model = hmm.GaussianHMM(n_components=3, covariance_type="full")
  model.fit(features)
  regime = model.predict(features)  # 0=trending, 1=choppy, 2=volatile
  Limitation: doesn't detect breakouts in real-time, states are post-hoc labels.

  2. Non-lagging classical indicators (underrated)

  Before going deep learning, these are surprisingly hard to beat as regime filters:

  - Choppiness Index — explicitly designed for range vs trend, non-lagging:
  CI = 100 × log10(Σ ATR(1,n) / (HH_n - LL_n)) / log10(n)
  - Values: >61.8 = choppy, <38.2 = strongly trending. Directly usable as a gate.
  - Rolling Hurst Exponent — H > 0.5 trending, H < 0.5 mean-reverting, H ≈ 0.5 random walk. Computable in ~10 lines with
   numpy. This is arguably the single most informative regime indicator.
  - ADX — measures trend strength without direction. Non-lagging unlike MAs. ADX > 25 = trending, < 20 = range-bound.

  3. Change-point detection — ruptures

  Detects when the character changes, not what the character is:
  import ruptures as rpt
  algo = rpt.Pelt(model="rbf").fit(price_series)
  breakpoints = algo.predict(pen=3)
  Good for detecting the breakout moment you described. Blind to the direction.

  4. Matrix Profile — stumpy

  Detects anomalous subsequences ("discords") and recurring patterns ("motifs") in time series with no labeled training
  data. Can flag when current price behavior is unlike historical norms — i.e., regime change:
  import stumpy
  mp = stumpy.stump(ts, m=window)  # matrix profile
  # high mp[:,0] values = discord = potential regime transition

  5. Lopez de Prado's meta-labeling — mlfinlab

  This is the most conceptually aligned with what you have. The idea: train a second model that predicts when your
  primary model (PatchTST) is likely to be right, using regime features as inputs. It's literally a "trust the signal"
  filter.
  # pip install mlfinlab
  from mlfinlab.feature_engineering.feature_matrix import get_orthogonal_features
  His book Advances in Financial Machine Learning has the full pipeline. mlfinlab implements the building blocks. It's
  dense but there are worked examples.

  ---
  The most pragmatic path for your setup

  Rather than a separate NN, you could feed regime features directly into PatchTST as additional unique_id series — your
   architecture already supports multivariate input:

  # Add these as extra series alongside KO_price, KO_vol, etc.:
  # KO_hurst    — rolling 60-day Hurst exponent
  # KO_chop     — Choppiness Index
  # KO_adx      — ADX
  # KO_breakout — rolling z-score of price vs N-day range (breakout strength)

  PatchTST would then learn to condition its predictions on the current regime rather than always predicting the same
  way regardless of market character. This is the lowest-friction integration given what you already have.

  ---
  The honest research landscape

  The papers you've been reading likely reference approaches that:
  - Use LSTMs or TCNs trained on labeled regime data (labels are themselves subjective/synthetic)
  - Use autoencoders to detect anomalous latent representations
  - Use reinforcement learning to adaptively weight regime features

  But almost none have public, runnable code. The gap between "published paper" and "testable repo" in this domain is
  enormous — most groups keep the working implementations proprietary. The closest public exception is the Lopez de
  Prado ecosystem (mlfinlab, mlfinlab-community).

  My honest recommendation for your immediate next step: add rolling Hurst + Choppiness Index as input series to your
  existing model and re-train. It's one afternoon of work with the infrastructure you have, and it directly addresses
  the "is this market trending or oscillating?" question that your trailing stop empirically revealed matters
  enormously.
