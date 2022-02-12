import pandas as pd
from scipy.stats import norm
import numpy as np
import math
import time
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import streamlit as st

number_assets = int(st.number_input('Enter number of assets'))
# breaker1 = False
# breaker2 = False


if number_assets > 0:
    x = []
    for a in range(1, number_assets+1):
        x.append(a)
        
    dct = {}
    for i in x:
        dct['row_%s' % i] = []
        
    for j in range (1, len(dct)+1):
        for y in range(1, len(dct)+1):
            text1 = 'Enter element {} for variance-covariance matrix row {}'.format(y, j)
            user_input = st.number_input(text1, format='%.5f')
            dct['row_%s' % j].append(user_input)

    covariance_matrix = pd.DataFrame({})
    for g in range(1, number_assets+1):
        df = pd.DataFrame(dct['row_%s' % g]).T
        covariance_matrix = pd.concat([covariance_matrix, df])
    
    if st.checkbox('Generate inverse covariance matrix'):
        inverse_cov_matrix = pd.DataFrame(np.linalg.inv(covariance_matrix), columns = covariance_matrix.columns, index = covariance_matrix.index)
    mode = st.selectbox('Select mode',    ('Tracking error', 'Portfolio optimization (no RL asset)', 'Portfolio optimization (RL asset)'))


    if mode == 'Tracking error':
        idx_vol = st.number_input('Enter index volatility', format='%.5f')
        idx_vols = []
        for x in range(0, number_assets):
            idx_vols.append(idx_vol)
        idx_vols_array = np.asarray([idx_vols])

        cor_coef = np.empty(number_assets, dtype = float)
        for i in range(0, number_assets):
            text2 = 'Enter correlation coefficient for asset {}'.format(i+1)
            cor_co = st.number_input(text2)
            cor_coef[i] = cor_co

        volatilities = np.empty(number_assets, dtype = float)
        for i in range(0, number_assets):
            text = 'Enter volatility for asset {}'.format(i+1)
            vol = st.number_input(text)
            volatilities[i] = vol

        def calc_unit(inverse_cov_matrix):
            unit = []
            for x in range(0, len(inverse_cov_matrix)):
                unit.append(1)
            return unit

        def calc_unit_transp(unit):
            unit_transp = pd.DataFrame({'test': unit})
            return unit_transp

        def calc_corr_vect(cor_coef, volatilities, idx_vols_array):
            corr_vect = pd.DataFrame((cor_coef*volatilities*idx_vols_array).tolist()[0])
            return corr_vect

        def calc_c(inverse_cov_matrix):
            unit = []
            for x in range(0, len(inverse_cov_matrix)):
                unit.append(1)
            unit_transp = pd.DataFrame({'test': unit})
            c = np.dot(np.dot(unit, inverse_cov_matrix), unit_transp)
            c = round(float(c),4)
            return c

        def calc_g(cor_coef, volatilities, idx_vols_array):
            g = np.dot(calc_unit(inverse_cov_matrix), pd.DataFrame(np.dot(inverse_cov_matrix, calc_corr_vect(cor_coef, volatilities, idx_vols_array)))).tolist()[0]    
            g = round(float(g),4)
            return g

        def calc_x_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix):
            g = calc_g(cor_coef, volatilities, idx_vols_array)
            c = calc_c(inverse_cov_matrix)
            unit = calc_unit(inverse_cov_matrix)
            inverse_unit = pd.DataFrame(np.dot(unit, inverse_cov_matrix))
            cor_part = pd.DataFrame(np.dot(inverse_cov_matrix, calc_corr_vect(cor_coef, volatilities, idx_vols_array)))
            x_te = (((1-g)/c)*inverse_unit)+cor_part
            return x_te

        def calc_var_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix, covariance_matrix):
            x_te = calc_x_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix)
            x_te_t = x_te.T
            var_te = np.dot(x_te_t, pd.DataFrame(np.dot(covariance_matrix, calc_x_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix)))).tolist()[0][0]
            var_te = round(var_te,4)
            return var_te

        def calc_cov_idx_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix):
            corr_vect_T = calc_corr_vect(cor_coef, volatilities, idx_vols_array).T
            x_te = calc_x_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix)
            cov_idx_te = np.dot(corr_vect_T, x_te).tolist()[0][0]
            cov_idx_te = round(cov_idx_te,4)
            return cov_idx_te

        def calc_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix, covariance_matrix):
            var_te = calc_var_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix, covariance_matrix)
            idx_vol = idx_vols_array.tolist()[0][0]
            cov_idx_te = calc_cov_idx_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix)
            te = math.sqrt(var_te + (idx_vol*idx_vol) - 2*cov_idx_te)
            te = round(te,4)
            return te

        # if breaker2 == True:
        def run():
            col1, col2 = st.columns(2)
            col1.metric('Tracking Error', calc_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix, covariance_matrix), delta=None, delta_color="normal")
            col2.metric('Tracking Error Portfolio Variance', calc_var_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix, covariance_matrix))
            st.subheader('Covariance Matrix', anchor=None)
            st.table(data=covariance_matrix)
            st.subheader('TE Portfolio Composition', anchor=None)
            st.table(data=calc_x_te(cor_coef, volatilities, idx_vols_array, inverse_cov_matrix))     

        run()

    if mode == 'Portfolio optimization (no RL asset)':
        returns = np.empty(number_assets, dtype = float)
        for i in range(0, number_assets):
            text = 'Enter return for asset {}'.format(i+1)
            return_x = st.number_input(text)
            returns[i] = return_x
        returns = pd.DataFrame(returns)
        returns.columns = ['returns']
        expected_return = float(st.number_input('Enter expected return'))
        shortfall_prob = float(st.number_input('Enter desired shortfall probability'))

        def calc_unit(inverse_cov_matrix):
            unit = []
            for x in range(0, len(inverse_cov_matrix)):
                unit.append(1)
            unit_transp = pd.DataFrame({'test': unit})
            return unit

        def calc_unit_transp(unit):
            unit_transp = pd.DataFrame({'test': unit})
            return unit_transp

        def calc_a(returns, inverse_cov_matrix):
            unit = calc_unit(inverse_cov_matrix)
            inverse_unit = np.dot(unit, inverse_cov_matrix)
            a = np.dot(returns.returns, inverse_unit)
            return float(a)

        def calc_b(returns, inverse_cov_matrix):
            b = np.dot(np.dot(returns.returns, inverse_cov_matrix), returns)
            return float(b)

        def calc_c(inverse_cov_matrix):
            unit = []
            for x in range(0, len(inverse_cov_matrix)):
                unit.append(1)
            unit_transp = pd.DataFrame({'test': unit})
            c = np.dot(np.dot(unit, inverse_cov_matrix), unit_transp)
            return float(c)

        def calc_d(returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            d = (b*c) - a*a
            return float(d)

        def calc_x_min_mvp(inverse_cov_matrix):
            c = calc_c(inverse_cov_matrix)
            unit = calc_unit(inverse_cov_matrix)
            inverse_unit = np.dot(unit, inverse_cov_matrix)
            x_min_mvp = pd.DataFrame(((1/c)* inverse_unit))
            x_min_mvp.columns = ['mvp_weights']
            x_min_mvp.index = inverse_cov_matrix.index
            return x_min_mvp

        def calc_return_min(returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            return_min = a/c
            return return_min

        def calc_variance_min(inverse_cov_matrix):
            c = calc_c(inverse_cov_matrix)
            variance_min = 1/c
            return variance_min

        def calc_x_exp_re(returns, inverse_cov_matrix, expected_return):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            d = calc_d(returns, inverse_cov_matrix)
            
            one = (b-(expected_return*a))/d
            two = ((expected_return*c)-a)/d
            
            unit = calc_unit(inverse_cov_matrix)
            inverse_unit = np.dot(unit, inverse_cov_matrix)
            covar_ret = np.dot(returns.returns, inverse_cov_matrix)
            
            x_exp_re = pd.DataFrame(np.dot(one, inverse_unit) + np.dot(two, covar_ret))
            x_exp_re.columns = ['expected_return_weights']

            return x_exp_re

        def calc_variance_exp_re(returns, inverse_cov_matrix, expected_return):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            d = calc_d(returns, inverse_cov_matrix)
            
            variance_exp_re = (b - ((2*expected_return)*a) + ((expected_return*expected_return)*c))/d
            return variance_exp_re

        def calc_x_tang(returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            covar_ret = np.dot(returns.returns, inverse_cov_matrix)

            x_tang = pd.DataFrame(1/a * covar_ret)
            x_tang.columns = ['tangent_portfolio_weights']
            return x_tang

        def calc_return_tang(returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            return_tang = b/a
            return float(return_tang)

        def calc_var_tang(returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            var_tang = b/(a*a)
            return float(var_tang)

        def calc_beta_no_riskless(returns, inverse_cov_matrix):
            c = calc_c(inverse_cov_matrix)
            d = calc_d(returns, inverse_cov_matrix)
            beta = c/d
            return beta

        def calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            s = norm.ppf(1-shortfall_prob)
            beta = calc_beta_no_riskless(returns, inverse_cov_matrix)
            return_min = calc_return_min(returns, inverse_cov_matrix)
            var_min = calc_variance_min(inverse_cov_matrix)
            one = 1 - (beta * (s**2))
            two = 2 * beta * s * return_min
            three = -var_min - (beta * (return_min**2))
            var1 = ((-two + np.sqrt(two**2 - (4 * one * three))) / (2 * one))
            ret1 = var1*s
            var2 = ((-two - np.sqrt(two**2 - (4 * one * three))) / (2 * one))
            ret2 = var2*s
            return var1, ret1, var2, ret2, s

        def plot_eff_frontier(returns, inverse_cov_matrix, shortfall_prob):
            x = np.linspace(0,1,100)
            a = calc_a(returns, inverse_cov_matrix)    
            c = calc_c(inverse_cov_matrix)
            beta = calc_beta_no_riskless(returns, inverse_cov_matrix)
            y = np.sqrt(1/c + beta*(x-(a/c))**2)
            
            fig = plt.figure(figsize=(15,12))
            ax = fig.add_subplot(1, 1, 1)

            plt.plot(y,x, 'b')
            plt.text(math.sqrt(calc_variance_min(inverse_cov_matrix)), calc_return_min(returns, inverse_cov_matrix), 'Min-var portfolio', horizontalalignment='left')
            plt.plot(math.sqrt(calc_variance_min(inverse_cov_matrix)), calc_return_min(returns, inverse_cov_matrix),'ko')

            plt.text(math.sqrt(calc_var_tang(returns, inverse_cov_matrix)), calc_return_tang(returns, inverse_cov_matrix), 'Tangent portfolio', horizontalalignment='left')
            plt.plot(math.sqrt(calc_var_tang(returns, inverse_cov_matrix)), calc_return_tang(returns, inverse_cov_matrix), 'ko')

            plt.text(np.sqrt(1/c + beta*(expected_return-(a/c))**2), expected_return, 'Preferred allocation', horizontalalignment='right')
            plt.plot(np.sqrt(1/c + beta*(expected_return-(a/c))**2), expected_return, 'ko')

            plt.text(calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[0], calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[1], 'P1', horizontalalignment='right')
            plt.plot(calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[0], calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[1], 'ko') 
            
            plt.text(calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[2], calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[3], 'P2', horizontalalignment='right')
            plt.plot(calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[2], calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[3], 'ko')         
            
            plt.xlabel('Standard Deviation Portfolio Returns')
            plt.ylabel('Expected Portfolio Returns')
            plt.title('Efficient Frontier')

        # if breaker2 == True:
        def run():
            st.subheader('Risk Efficient Frontier', anchor=None)
            st.pyplot(fig=plot_eff_frontier(returns, inverse_cov_matrix, shortfall_prob))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write('Efficient frontier equation:', round(calc_variance_min(inverse_cov_matrix),5), '+', round(calc_beta_no_riskless(returns, inverse_cov_matrix),5), '(portfolio return - ', round(calc_return_min(returns, inverse_cov_matrix),5), ')^2')

            st.subheader('Building Blocks', anchor=None)
            col1, col2, col3 = st.columns(3)
            col1.metric("A", round(calc_a(returns, inverse_cov_matrix),5))
            col2.metric("B", round(calc_b(returns, inverse_cov_matrix),5))
            col3.metric("C", round(calc_c(inverse_cov_matrix),5))
            col4, col5 = st.columns(2)
            col4.metric("D", round(calc_d(returns, inverse_cov_matrix),5))
            col5.metric("Beta", round(calc_beta_no_riskless(returns, inverse_cov_matrix),5))

            st.subheader('Min-Var Portfolio', anchor=None)
            col6, col7 = st.columns(2)
            col6.metric('Variance', round(calc_variance_min(inverse_cov_matrix),5))
            col7.metric('Return', round(calc_return_min(returns, inverse_cov_matrix),5))
            st.table(data=calc_x_min_mvp(inverse_cov_matrix))

            st.subheader('Tangent Portfolio', anchor=None)
            col8, col9 = st.columns(2)
            col8.metric('Variance', round(calc_var_tang(returns, inverse_cov_matrix),5))
            col9.metric('Return', round(calc_return_tang(returns, inverse_cov_matrix),5))
            st.table(data=calc_x_tang(returns, inverse_cov_matrix))

            st.subheader('Expected Return Portfolio', anchor=None)
            col10, col11 = st.columns(2)
            col10.metric('Variance', round(calc_variance_exp_re(returns, inverse_cov_matrix, expected_return),5))
            col11.metric('Return', round(expected_return,5))
            st.table(data=calc_x_exp_re(returns, inverse_cov_matrix, expected_return)) 

            st.subheader('Shortfall Probability', anchor=None)
            col12, col13, col14, col15 = st.columns(4)
            if calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[0] > calc_return_min(returns, inverse_cov_matrix):
                sf_return = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[1]
                sf_variance = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[0]
            else: 
                sf_return = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[3]
                sf_variance = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[2]    
            col12.metric('Standard Deviation', round(sf_variance,5))
            col13.metric('Return', round(sf_return,5))
            col14.metric('Shortfall Probability', round(shortfall_prob, 5))
            col15.metric('S', round(calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[4],2))
            st.subheader('Covariance Matrix', anchor=None)
            st.table(data=covariance_matrix)
        run()

    if mode == 'Portfolio optimization (RL asset)':
        returns = np.empty(number_assets, dtype = float)
        for i in range(0, number_assets):
            text = 'Enter return for asset {}'.format(i+1)
            return_x = st.number_input(text)
            returns[i] = return_x
        returns = pd.DataFrame(returns)
        returns.columns = ['returns']
        expected_return = float(st.number_input('Enter expected return'))
        shortfall_prob = float(st.number_input('Enter desired shortfall probability'))
        riskless_return = float(st.number_input('Enter riskless return'))

        def calc_unit(inverse_cov_matrix):
            unit = []
            for x in range(0, len(inverse_cov_matrix)):
                unit.append(1)
            unit_transp = pd.DataFrame({'test': unit})
            return unit

        def calc_unit_transp(unit):
            unit_transp = pd.DataFrame({'test': unit})
            return unit_transp

        def calc_pi(returns, riskless_return):
            pi = returns - riskless_return
            return pi

        def calc_a(returns, inverse_cov_matrix):
            unit = calc_unit(inverse_cov_matrix)
            inverse_unit = np.dot(unit, inverse_cov_matrix)
            a = np.dot(returns.returns, inverse_unit)
            return float(a)

        def calc_b(returns, inverse_cov_matrix):
            b = np.dot(np.dot(returns.returns, inverse_cov_matrix), returns)
            return float(b)

        def calc_c(inverse_cov_matrix):
            unit = []
            for x in range(0, len(inverse_cov_matrix)):
                unit.append(1)
            unit_transp = pd.DataFrame({'test': unit})
            c = np.dot(np.dot(unit, inverse_cov_matrix), unit_transp)
            return float(c)

        def calc_d(returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            d = (b*c) - a*a
            return float(d)

        def calc_f(returns, riskless_return, inverse_cov_matrix):
            pi = calc_pi(returns, riskless_return)
            returns_t = returns.T
            f = np.dot(pi.T, np.dot(inverse_cov_matrix, pi)).tolist()[0][0]
            return f

        def calc_beta(returns, riskless_return, inverse_cov_matrix):
            f = calc_f(returns, riskless_return, inverse_cov_matrix)
            beta = 1 / f
            return beta

        def calc_beta_no_riskless(returns, inverse_cov_matrix):
            c = calc_c(inverse_cov_matrix)
            d = calc_d(returns, inverse_cov_matrix)
            beta_no_riskless = c/d
            return beta_no_riskless

        def calc_k(riskless_return):
            a = calc_a(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            k = 1 / (a - (c * riskless_return))
            return k 

        def calc_x_min_mvp(inverse_cov_matrix):
            c = calc_c(inverse_cov_matrix)
            unit = calc_unit(inverse_cov_matrix)
            inverse_unit = np.dot(unit, inverse_cov_matrix)
            x_min_mvp = pd.DataFrame(((1/c)* inverse_unit))
            x_min_mvp.columns = ['mvp_weights']
            x_min_mvp.index = inverse_cov_matrix.index
            return x_min_mvp

        def calc_return_min(returns, inverse_cov_matrix):
            a = calc_a(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            return_min = a/c
            return return_min

        def calc_variance_min(inverse_cov_matrix):
            c = calc_c(inverse_cov_matrix)
            variance_min = 1/c
            return variance_min

        def calc_x_exp_re(returns, inverse_cov_matrix, expected_return, riskless_return):
            pi = calc_pi(returns, riskless_return)    
            f = calc_f(returns, riskless_return, inverse_cov_matrix)
            x_exp_re = pd.DataFrame(((expected_return - riskless_return) / f) * np.dot(inverse_cov_matrix, pi))
            x_exp_re.columns = ['expected_return_weights']
            return x_exp_re

        def calc_variance_exp_re(returns, inverse_cov_matrix, expected_return, riskless_return):
            f = calc_f(returns, riskless_return, inverse_cov_matrix)
            variance_exp_re = ((expected_return - riskless_return)**2) / f
            return variance_exp_re

        def calc_x_tang(returns, inverse_cov_matrix, riskless_return):
            pi = calc_pi(returns, riskless_return)
            k = calc_k(riskless_return)
            x_tang = pd.DataFrame(np.dot(k, np.dot(inverse_cov_matrix, pi)))
            x_tang.columns = ['tangent_portfolio_weights']
            return x_tang

        def calc_return_tang(returns, inverse_cov_matrix, riskless_return):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)
            return_tang = (b - (a * riskless_return)) / (a - (c * riskless_return))
            return float(return_tang)

        def calc_var_tang(returns, inverse_cov_matrix, riskless_return):
            a = calc_a(returns, inverse_cov_matrix)
            b = calc_b(returns, inverse_cov_matrix)
            c = calc_c(inverse_cov_matrix)    
            var_tang = (b - (2*a*riskless_return) + (c *(riskless_return)**2)) / (a - (c * riskless_return))**2
            return float(var_tang)

        def plot_eff_frontier(returns, inverse_cov_matrix, riskless_return):
            beta = calc_beta(returns, riskless_return, inverse_cov_matrix)
            x = np.linspace(0,1,100)
            y1 = np.sqrt(beta*(x-(riskless_return))**2)
            
            beta_no_riskless = calc_beta_no_riskless(returns, inverse_cov_matrix)
            a = calc_a(returns, inverse_cov_matrix)    
            c = calc_c(inverse_cov_matrix)
            y2 = np.sqrt(1/c + beta_no_riskless*(x-(a/c))**2)
            
            fig = plt.figure(figsize=(15,12))
            ax = fig.add_subplot(1, 1, 1)

            plt.plot(y1,x, 'b')
            plt.plot(y2, x, 'r')
            plt.text(math.sqrt(calc_variance_min(inverse_cov_matrix)), calc_return_min(returns, inverse_cov_matrix), 'Min-var portfolio', horizontalalignment='left')
            plt.plot(math.sqrt(calc_variance_min(inverse_cov_matrix)), calc_return_min(returns, inverse_cov_matrix),'ko')

            plt.text(math.sqrt(calc_var_tang(returns, inverse_cov_matrix, riskless_return)), calc_return_tang(returns, inverse_cov_matrix, riskless_return), 'Tangent portfolio', horizontalalignment='left')
            plt.plot(math.sqrt(calc_var_tang(returns, inverse_cov_matrix, riskless_return)), calc_return_tang(returns, inverse_cov_matrix, riskless_return), 'ko')

            plt.text(np.sqrt(calc_variance_exp_re(returns, inverse_cov_matrix, expected_return, riskless_return)), expected_return, 'Preferred allocation', horizontalalignment='right')
            plt.plot(np.sqrt(calc_variance_exp_re(returns, inverse_cov_matrix, expected_return, riskless_return)), expected_return, 'ko')

            plt.xlabel('Standard Deviation Portfolio Returns')
            plt.ylabel('Expected Portfolio Returns')
            plt.title('Efficient Frontier')

        # if breaker2 == True:
        def run():
            st.subheader('Risk Efficient Frontier', anchor=None)
            st.pyplot(fig=plot_eff_frontier(returns, inverse_cov_matrix, shortfall_prob))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write('Riskfree efficient frontier equation:', round(calc_beta(returns, riskless_return, inverse_cov_matrix),5), '(portfolio return - ', riskless_return, ')^2')
            st.write('Efficient frontier equation:', round(calc_variance_min(inverse_cov_matrix),5), '+', round(calc_beta_no_riskless(returns, inverse_cov_matrix),5), '(portfolio return - ', round(calc_return_min(returns, inverse_cov_matrix),5), ')^2')


            st.subheader('Building Blocks', anchor=None)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("A", round(calc_a(returns, inverse_cov_matrix),5))
            col2.metric("B", round(calc_b(returns, inverse_cov_matrix),5))
            col3.metric("C", round(calc_c(inverse_cov_matrix),5))
            col4.metric("D", round(calc_d(returns, inverse_cov_matrix),5))
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("F", round(calc_f(returns, riskless_return, inverse_cov_matrix),5))
            col6.metric("Beta", round(calc_beta(returns, riskless_return, inverse_cov_matrix),5))
            col7.metric("Beta without riskless asset", round(calc_beta_no_riskless(returns, inverse_cov_matrix),5))
            col8.metric("K", round(calc_k(riskless_return),5))

            st.subheader('Min-Var Portfolio', anchor=None)
            col9, col10 = st.columns(2)
            col9.metric('Variance', round(calc_variance_min(inverse_cov_matrix),5))
            col10.metric('Return', round(calc_return_min(returns, inverse_cov_matrix),5))
            st.table(data=calc_x_min_mvp(inverse_cov_matrix))
            st.table(data=calc_pi(returns, riskless_return))

            st.subheader('Tangent Portfolio', anchor=None)
            col11, col12 = st.columns(2)
            col11.metric('Variance', round(calc_var_tang(returns, inverse_cov_matrix, riskless_return),5))
            col12.metric('Return', round(calc_return_tang(returns, inverse_cov_matrix, riskless_return),5))
            st.table(data=calc_x_tang(returns, inverse_cov_matrix, riskless_return))

            st.subheader('Expected Return Portfolio', anchor=None)
            col13, col14 = st.columns(2)
            col13.metric('Variance', round(calc_variance_exp_re(returns, inverse_cov_matrix, expected_return, riskless_return),5))
            col14.metric('Return', round(expected_return,5))
            st.table(data=calc_x_exp_re(returns, inverse_cov_matrix, expected_return, riskless_return)) 

            # st.subheader('Shortfall Probability', anchor=None)
            # col12, col13, col14, col15 = st.columns(4)
            # if calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[0] > calc_return_min(returns, inverse_cov_matrix):
            #     sf_return = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[1]
            #     sf_variance = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[0]
            # else: 
            #     sf_return = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[3]
            #     sf_variance = calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[2]    
            # col12.metric('Variance', round(sf_variance,5))
            # col13.metric('Return', round(sf_return,5))
            # col14.metric('Shortfall Probability', round(shortfall_prob, 5))
            # col15.metric('S', round(calc_shortfall_prob(shortfall_prob, returns, inverse_cov_matrix)[4],2))

            st.subheader('Covariance Matrix', anchor=None)
            st.table(data=covariance_matrix)
        run()