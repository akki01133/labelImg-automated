
import os
folder_path = './images/train/labels'
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            lines = f.readlines()
            updated_lines = []
            for line in lines:
                line = line.split()
                if line[0] == '1':
                    line[0] = '0'
                line = ' '.join(line)
                updated_lines.append(line)
        with open(os.path.join(folder_path, filename), mode='w') as f:
            for line in updated_lines:
                if line is not updated_lines[-1]:
                    f.write(line + '\n')
                else:
                    f.write(line)
