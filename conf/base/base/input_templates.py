reduc1_inp_script= """%maxcore {maxcore}
# {filename}
  ! {shell} {calc_type} {functional} {basis_set} {aux_basis_set} {convergence_criteria} {additional_keywords} {%mdci} {%pal}
  %scf
   MaxIter 500
  end
  * xyzfile {charge} {spin} {geom_file}
  """

inp_reduc1= """{molecule_name}
{param_set_id} {comment}
{n_params_to_fit}
{indexes_params_to_fit}
{molar_mass} {charachteristic_temperature} {segment_volume} {number_of_segments} {abs_quadrupole_moment} {abs_dipole_moment} {polarizability} {assoc_energy} {assoc_volume} {assoc_scheme}
{n_data_points}
{fitting_routine}
{data}
"""
#old example
"""CC1C(C)OCO1
1 VLE and Density data from OMT group
5
1 2 3 7 8
102.142 284.850551152188 15.1987655845569 2.90259340083117 0 1.5339859647337 0 600 0.04 2
19
101
VLE 1 0.00701307317073171 293.45 -1
VLE 1 0.0422039498432602 338.15 -1
VLE 1 0.119350193548387 368.15 -1
VLE 1 0.0070 293.45 105.921270947403
VLE 1 0.0085 298.15 106.454470604175
VLE 1 0.0104 303.15 -1
VLE 1 0.0127 308.15 -1
VLE 1 0.0156 313.15 108.288452567745
VLE 1 0.0191 318.15 -1
VLE 1 0.0235 323.15 -1
VLE 1 0.0285 328.15 110.291434062908
VLE 1 0.0346 333.15 -1
VLE 1 0.0422 338.15 -1
VLE 1 0.0511 343.15 112.272332567572
VLE 1 0.0611 348.15 -1
VLE 1 0.0727 353.15 -1
VLE 1 0.0866 358.15 114.448664941118
VLE 1 0.1021 363.25 -1
VLE 1 0.1194 368.15 -1
"""