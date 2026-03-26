import sys
sys.path.insert(0, r'C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\validation')
exec(open(r'C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\validation\pna_solver_validate_v5.py', encoding='utf-8').read())

data = create_test_data()
validate_forward(data)
validate_backward(data, n_check=6)
