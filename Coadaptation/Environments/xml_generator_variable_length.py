#xml_generator
import os
import shutil
from sys import argv
import numpy as np
import time

def generate_xml(param):
    xml_models_path = "Environments/xml_models_variable_length"
    template_filename = "snake_v14_template_variable_length.xml"

    # if len(argv) != 9:
    #     print("ERROR: needed 8 values" )

    link_sizes = param
    # for i in range(8):
    #     link_sizes.append(int(argv[i+1]))
    # link_sizes = [10, 10, 10, 10, 100, 100, 100, 100]
    link_size_string = str(link_sizes)
    link_size_string=link_size_string.replace("[","")
    link_size_string=link_size_string.replace("]","")
    link_size_string = '_'.join(link_size_string.split())
    link_size_string=link_size_string.replace("   ","_")
    link_size_string=link_size_string.replace("  ","_")
    link_size_string=link_size_string.replace(" ","_")
    # print("link_size_string",link_size_string)

    # modified_file = f"{xml_models_path}/snake_v14_{link_size_string}.xml"         #name of new file
    # shutil.copy2(template_filename, xml_models_path)                        #copy template as new file
    # os.rename(f"{xml_models_path}/{template_filename}",modified_file)       #rename template as new file name

    modified_file = f"Environments/xml_models_variable_lenght/snake_v14_{link_size_string}.xml"         #name of new file
    shutil.copy("Environments/snake_v14_template_variable_length.xml", modified_file)                        #copy template as new file
    


    idx = 0
    idx_next = 0

    for i in range(len(link_sizes)):
        idx = link_sizes[i]
        if i != len(link_sizes)-1:
            idx_next = link_sizes[i+1]
        else:
            idx_next = idx
        
        with open(modified_file, 'r') as template_file:
            file_content_template = template_file.read()
            template_file.close()
        new_file_content = file_content_template

        #modified values
        layer_0_1_size = str(np.round(0.0300 + 0.0005*idx,4))
        arm_pos = str(np.round(-0.1130 - 0.0005*idx,4))
        back_pos = str(np.round(0.0150 + 0.0005*idx,4))
        motor_mesh_pos = str(np.round(0.003 + 0.0005*idx,4))
        caps1st_value = str(np.round(-0.0420 + 0.0005*idx,4))
        caps2nd_value = str(np.round(-0.0910 - 0.0005*idx,4))
        slider_rot_pos = str(np.round(-0.0350 + 0.0005*idx,4))
        motor_joint_pos = str(np.round(0.0025 + 0.0005*idx,4))


        body_seg_pos1 = np.round(-0.1270 - 0.001*idx,4)
        body_seg_pos2 = np.round(-0.1270 - 0.001*(idx_next),4)
        body_seg_pos = str(np.round((body_seg_pos1 + body_seg_pos2)/2,4))


        mod1 = new_file_content.replace(f"layer_0_1_size_{i+1}", layer_0_1_size)
        mod2 = mod1.replace(f"arm_pos_{i+1}", arm_pos)
        mod3 = mod2.replace(f"back_pos_{i+1}", back_pos)
        mod4 = mod3.replace(f"motor_mesh_pos_{i+1}", motor_mesh_pos)
        mod5 = mod4.replace(f"caps1st_value_{i+1}", caps1st_value)
        mod6 = mod5.replace(f"caps2nd_value_{i+1}", caps2nd_value)
        mod7 = mod6.replace(f"body_seg_pos_{i+1}", body_seg_pos)
        mod8 = mod7.replace(f"slider_rot_pos_{i+1}", slider_rot_pos)
        mod9 = mod8.replace(f"motor_joint_pos_{i+1}", motor_joint_pos)

        with open(modified_file, 'w') as template_file:
            template_file.write(mod9)

        
    print("sleep for 2 sec bc generating xml")
    time.sleep(2)        

    print("Generated file with lengths: ",link_sizes)