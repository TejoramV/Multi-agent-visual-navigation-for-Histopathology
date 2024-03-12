patch_number_to_zoom_lvl_probabilities = [0.00036146399490007163, 0.056983155777837656, 0.019962670627435775, 0.1151657148114801, 0.029827351651890457, 0.027218238815975394, 0.029084707444186675, 0.027964169059996453, 0.02540434676884049, 0.18809929087336275, 0.02383033537286654, 0.023337429925275533, 0.019726076012592093, 0.013015989852719852, 0.014396125105974671, 0.013604190353511786, 0.013896647585749117, 0.010170282401961106, 0.011524129364677739, 0.13916692407284487, 0.007097838445310497, 0.008638989478111711, 0.006013446460610283, 0.004255417030869025, 0.004587306698913636, 0.004156835941350824, 0.0036967908569325507, 0.004087829178688083, 0.003801944019085299, 0.0034010475883779465, 0.0024579551653204873, 0.00291800024973876, 0.0017843177202794445, 0.0020964911704204154, 0.002109635315689509, 0.0032860363172733786, 0.0019387614271912934, 0.002214788477842257, 0.002385662366340473, 0.09482843604387516, 0.0012684100184675241, 0.0014721442701384735, 0.0028424214144414724, 0.0011665428926320495, 0.0006572072634546756, 0.0014294257980139197, 0.003949815653362601, 0.0007820766435110641, 0.000460045084418273, 0.0005783423918401146, 0.0003417477769964314, 0.00045347301178372626, 0.00024316668747823, 0.00030888741382369757, 0.0002891711959200573, 0.0005421959923501075, 0.001284840200053891, 0.0009168041325192726, 0.00015444370691184879, 0.01236206862558245]  

# todo bin (1,2)->1 (3,6)->5 (7,12)->10 (13-27)->20 (28 45)->40 (45<)->50
temp = []
temp.append(sum(patch_number_to_zoom_lvl_probabilities[:2]))
temp.append(sum(patch_number_to_zoom_lvl_probabilities[2:6]))
temp.append(sum(patch_number_to_zoom_lvl_probabilities[6:12]))
temp.append(sum(patch_number_to_zoom_lvl_probabilities[12:27]))
temp.append(sum(patch_number_to_zoom_lvl_probabilities[27:45]))
temp.append(sum(patch_number_to_zoom_lvl_probabilities[45:]))

print(temp)
print(sum(temp))
