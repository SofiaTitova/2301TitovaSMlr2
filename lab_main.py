import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def DCT_matrix_formula(N): #получаем квадратную матрицу ДКП
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                T[i, j] = 1 / np.sqrt(N)
            else:
                T[i, j] = np.sqrt(2 / N) * np.cos(np.pi * (2 * j + 1) * i / (2 * N))
    return T

def dct(image): #дискретное косинусное преобразование изображения
    height, width = image.shape
    C_h = DCT_matrix_formula(height)
    C_w = DCT_matrix_formula(width)
    use_DCT = np.dot(np.dot(C_h, image), C_w.T) ##C * H * C'
    return use_DCT.astype(np.int16)

def idct(DCT_K): # обаратное дискретное косинусное преобращование
    height, width = DCT_K.shape
    C_h = DCT_matrix_formula(height)
    C_w = DCT_matrix_formula(width)
    use_IDCT = np.dot(np.dot(C_h.T, DCT_K), C_w)
    return use_IDCT.astype(np.int16)

def separation_blocks(image, height, width): #разделение на квадратные блоки
    blocks = []
    gr = 8
    for i in range(0, height, gr):
        for j in range(0, width, gr):
            blocks.append(image[i:i+gr, j:j+gr])
    return np.array(blocks)

def into_one_im(blocks, height, width): #объединение на блоки
    image = np.zeros((height, width), dtype=int)
    ind = 0
    gr = 8
    for i in range(0, height, gr):
        for j in range(0, width, gr):
            image[i:i+gr, j:j+gr] = blocks[ind]
            ind += 1
    return image

def RGBtoYCbCr(A):
    R, G, B = A[:, :, 0], A[:, :, 1], A[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.081312 * B + 128

    func_min = lambda x: min(x, 255)
    vect_func_min = np.vectorize(func_min)
    func_max = lambda x: max(x, 0)
    vect_func_max = np.vectorize(func_max)

    T = np.round(np.dstack((Y, Cb, Cr)))
    T = vect_func_min(T)
    T = vect_func_max(T)

    return T.astype(np.uint8)

def YCbCrtoRGB(Y, Cb, Cr):
    Y = Y.astype(np.int16)
    Cb = Cb.astype(np.int16)
    Cr = Cr.astype(np.int16)

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    R = np.uint8(np.clip(R, 0, 255))
    G = np.uint8(np.clip(G, 0, 255))
    B = np.uint8(np.clip(B, 0, 255))

    RGB_img = np.dstack((R, G, B))

    return RGB_img

def get_Qmatrix(quality_level):
    Qmatrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    if quality_level <= 50: #изменение в отношении с уровнем качества
        scale = 5000 / quality_level
    else:
        scale = 200 - 2 * quality_level
    matrix_with_scale = np.floor((Qmatrix * scale + 50) / 100)
    matrix_with_scale = np.clip(matrix_with_scale, 1, 255)
    return matrix_with_scale.astype(np.uint8)

def quant_after_dct(use_DCT, Qmatrix):
    quantization_matrix = np.round(use_DCT / Qmatrix)
    return quantization_matrix.astype(int)

def quant_before_dct(use_DCT, Qmatrix):
    quantization_matrix = np.round(use_DCT * Qmatrix)
    return quantization_matrix.astype(int)

def reverse(A, i, j, k, col, res) :
    if (j >= 0 and k < col) :
        reverse(A, i, j - 1, k + 1, col, res)
        res.append(A[j][k])
        
def zigzag(A) :
    res = list()
    i = 0
    j = 0
    k = 0
    counter = 0
    row = len(A)
    col = len(A[0])
    #верхняя половина матрицы
    while (i < row) :
        if (counter % 2 == 0) : #слева направо
            j = 0
            while (j <= i and j < col and i - j >= 0) :
                res.append(A[i - j][j])
                j += 1
        else :
            #справа налево
            j = i
            k = 0
            while (j >= 0 and j < col and k <= i) :
                res.append(A[k][j])
                j -= 1
                k += 1
        #следующая строка
        i += 1
        counter += 1
    #ниже главной диагонали
    i = 1
    while (i < col) :
        if (counter % 2 == 0) :
            #аслева направо
            j = row - 1
            k = i
            while (j >= 0 and k < col) :
                res.append(A[j][k])
                j -= 1
                k += 1
        else :
            reverse(A, i, row - 1, i, col, res)
        counter += 1
        i += 1
    return res

def zigzag_rev(arr, rows, cols):
    back_to_matrix = np.zeros((rows, cols), dtype=int)
    index = 0
    for i in range(1, rows + cols):
        start_col = max(0, i - rows)
        count = min(i, (cols - start_col), rows)
        for j in range(count):
            if i % 2 == 1:  # eсли диагональ сверху вниз
                back_to_matrix[min(rows, i) - j - 1][start_col + j] = arr[index]
            else:  # eсли диагональ снизу вверх
                back_to_matrix[start_col + j][min(rows, i) - j - 1] = arr[index]
            index += 1
    return back_to_matrix

def RLE(data):
    first = 0
    res = []
    k = 1
    cur_elem = ''
    if data[first] < 0:
        cur_elem += '𖼩'
    cur_elem += chr(np.abs(data[first]))

    for i in range(1, len(data)):
        if data[i] < 0:
            next_char = '𖼩' + chr(np.abs(data[i]))
        else:
            next_char = chr(data[i])
        if next_char == cur_elem:
            k += 1
        else:
            if k > 1:
                res.append('𖼣')
                res.append(chr(k))
            res.append(cur_elem)
            cur_elem = next_char
            k = 1

    if k > 1:
        res.append('𖼣')
        res.append(chr(k))
        res.append(cur_elem)
    return ''.join(res)

def deRLE(reverse):
    res = []
    i = 0
    length = len(reverse)
    while i < length:
        if reverse[i] == '𖼣':
            k = ord(reverse[i + 1])
            i += 2
        else:
            k = 1

        if reverse[i] == '𖼩':
            number = -ord(reverse[i + 1])
            i += 2
        else:
            number = ord(reverse[i])
            i += 1
        res.extend([number] * k)
    return np.array(res)

def choose_downsampling(type, image, M, N):
    match type:
        case 1: return downsampling_cr(image, M, N)
        case 2: return downsampling_mid(image, M, N)
        case 3: return downsampling_aprox(image, M, N)

def downsampling_cr(image, M, N):
    height, width = image.shape
    new_height= np.floor(height / M)
    new_width =np.floor(width / N)
    after_downsampling = np.zeros((new_height, new_width), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            after_downsampling[i, j] = image[int(i * M), int(j * N)]
    return after_downsampling

def downsampling_mid(image, M, N):
    height, width = image.shape
    new_height = int((height / M))
    new_width = int((width / N))
    height_of_block = int(np.floor(height / new_height))
    width_of_block = int(np.floor(width / new_width))
    after_downsampling = np.zeros((new_height, new_width), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            block = image[i * height_of_block: (i + 1) * height_of_block, j * width_of_block: (j + 1) * width_of_block]
            middle_color = np.mean(block)
            after_downsampling[i, j] = int(middle_color)
    return after_downsampling

def downsampling_aprox(image, M, N):
    height, width = image.shape
    new_height = int((height / M))
    new_width = int((width / N))
    height_of_block = int(np.floor(height / new_height))
    width_of_block = int(np.floor(width / new_width))
    after_downsampling = np.zeros((new_height, new_width), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            block = image[i * height_of_block: (i + 1) * height_of_block, j * width_of_block: (j + 1) * width_of_block]
            middle_in_block = np.mean(block)
            differences = np.abs(block - middle_in_block)
            min_index = np.unravel_index(np.argmin(differences, axis=None), differences.shape)
            aprox_color = block[min_index]

            after_downsampling[i, j] = int(aprox_color)
    return after_downsampling

def upsampling(image, N, M):
    height, width = image.shape
    new_height, new_width = int((height * M)), int((width * N))
    upsampling_image = np.zeros((new_height, new_width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            block = image[i,j]
            upsampling_image[i * M: (i + 1) * M, j * N: (j + 1) * N] = block
    return upsampling_image

def DCT(image):
    N, M = image.shape
    ratio = 1 / np.sqrt(2)
    dct_result = np.zeros((N, M), dtype=np.float64)
    for u in range(N):
        for v in range(M):
            sum = 0
            for x in range(N):
                for y in range(M):
                    sum += image[x, y] * np.cos(np.pi * u * (2 * x + 1) / (2 * N)) * np.cos(np.pi * v * (2 * y + 1) / (2 * M))
            if u == 0:
                c_u = ratio
            else:
                c_u = 1
            if v == 0:
                c_v = ratio
            else:
                c_v = 1
            dct_result[u, v] = 2 / np.sqrt(N * M) * c_u * c_v * sum
    return dct_result.astype(np.int16)

def IDCT(dct_ratio):
    N, M = dct_ratio.shape
    image = np.zeros((N, M), dtype=np.float64)
    ratio = 1 / np.sqrt(2)
    for x in range(N):
        for y in range(M):
            sum = 0.0
            for u in range(N):
                for v in range(M):
                    cu = 1.0
                    cv = 1.0
                    if u == 0:
                        cu = ratio
                    if v == 0:
                        cv = ratio
                    sum += cu * cv * dct_ratio[u, v] * np.cos(np.pi / (2.0 * N) * (2.0 * x + 1) * u) * np.cos(np.pi / (2.0 * M) * (2.0 * y + 1) * v)
            image[x, y] = sum * np.sqrt(2.0 / N) * np.sqrt(2.0 / M)

    return image.astype(np.int16)

def write_matrix_to_file(txt, file_):
    with open(file_, 'w') as file:
        for i in txt:
            for j in i:
                file.write(f"{j[0]} {j[1]} {j[2]} ")
            file.write("\n")

def to_file(file_name, first, second, third):
    with open(file_name, 'w') as file:
        file.write(' '.join(map(str, first)) + '\n')
        file.write(' '.join(map(str, second)) + '\n')
        file.write(' '.join(map(str, third)) + '\n')
        file.write("\n")

def red_img():
    width, height = 800, 800
    image = Image.new('RGB', (width, height), color='red')
    image.save('solid_color_image.png')
    return image

def random_img():
    width, height = 800, 800
    random_image_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(random_image_data)
    image.save('random_color_image.png')
    return image

def main_function(file_path):
    image = Image.open(file_path)
    rgb_img = image.convert("RGB")
    rgb_to_file = np.array(rgb_img)
    height, width = image.size
    write_matrix_to_file(rgb_to_file, 'rgb_matrix.txt')

    ycbcr_to_file = RGBtoYCbCr(rgb_to_file)
    write_matrix_to_file(ycbcr_to_file, 'ycbcr_matrix.txt')
    Y, Cb, Cr = ycbcr_to_file[:, :, 0], ycbcr_to_file[:, :, 1], ycbcr_to_file[:, :, 2]

    height_koef = int(input("Введите коэффициент сжатия по высоте: "))
    width_koef = int(input("Введите коэффициент сжатия по длине: "))


    d_sampl_koef = int(input("Выберите даунсемплинг 1-3: "))
    downs_for_Cb = choose_downsampling(d_sampl_koef, Cb, height_koef, width_koef)
    downs_for_Cr = choose_downsampling(d_sampl_koef, Cr, height_koef, width_koef)

    to_file('downsampled_matrix.txt', Y, downs_for_Cb, downs_for_Cr)

    blocks_for_Y = separation_blocks(Y, height, width)
    new_height, new_width = int(height/height_koef), int(width/height_koef)
    blocks_for_Cb = separation_blocks(downs_for_Cb,new_height, new_width)
    blocks_for_Cr = separation_blocks(downs_for_Cr,new_height, new_width)

    koef_Y = np.array([dct(part) for part in blocks_for_Y])
    koef_Cb = np.array([dct(part) for part in blocks_for_Cb])
    koef_Cr = np.array([dct(part) for part in blocks_for_Cr])

    DCT_Y = into_one_im(koef_Y,height, width)
    DCT_Cb = into_one_im(koef_Cb,new_height,new_width)
    DCT_Cr = into_one_im(koef_Cr,new_height,new_height)

    to_file('DCT.txt', DCT_Y, DCT_Cb, DCT_Cr)

    quality = int(input("Выберите параметр качества: "))
    Qmatrix = get_Qmatrix(quality)

    q_koef_Y = np.array([quant_after_dct(part, Qmatrix) for part in koef_Y])
    q_koef_Cb = np.array([quant_after_dct(part, Qmatrix) for part in koef_Cb])
    q_koef_Cr = np.array([quant_after_dct(part, Qmatrix) for part in koef_Cr])

    Y_back = into_one_im(q_koef_Y, height, width)
    Cb_back = into_one_im(q_koef_Cb,new_height,new_width)
    Cr_back = into_one_im(q_koef_Cr,new_height,new_width)

    to_file('quantization_image.txt', Y_back, Cb_back, Cr_back)

    zigzag_Y = zigzag(Y_back)
    zigzag_Cb = zigzag(Cb_back)
    zigzag_Cr = zigzag(Cr_back)

    to_file('zigzag.txt', zigzag_Y, zigzag_Cb, zigzag_Cr)

    rle_Y = RLE(zigzag_Y)
    rle_Cb = RLE(zigzag_Cb)
    rle_Cr = RLE(zigzag_Cr)

    rle_result = rle_Y + "𑅪" + rle_Cb + "𑅪" + rle_Cr
    with open("rle_decode.txt", 'w', encoding='utf-8') as file:
        file.write(rle_result)

    print("Ваше изображение сжато")

    print(f"___________________________________________\n\n")

    print("Декодирование:")

    rle_Y_rev = deRLE(rle_Y)
    rle_Cb_rev = deRLE(rle_Cb)
    rle_Cr_rev = deRLE(rle_Cr)

    zigzag_Y_rev = zigzag_rev(rle_Y_rev, height, width)
    zigzag_Cb_rev = zigzag_rev(rle_Cb_rev, new_height, new_width)
    zigzag_Cr_rev = zigzag_rev(rle_Cr_rev, new_height, new_width)

    blocks_Y_decode = separation_blocks(zigzag_Y_rev, height, width)
    blocks_Cb_decode = separation_blocks(zigzag_Cb_rev,new_height, new_width)
    blocks_Cr_decode = separation_blocks(zigzag_Cr_rev,new_height, new_width)

    q_Y_rev = quant_before_dct(blocks_Y_decode, Qmatrix)
    q_Cb_rev = quant_before_dct(blocks_Cb_decode, Qmatrix)
    q_Cr_rev = quant_before_dct(blocks_Cr_decode, Qmatrix)


    DCT_Y_rev = np.array([idct(part) for part in q_Y_rev])
    DCT_Cb_rev = np.array([idct(part) for part in q_Cb_rev])
    DCT_Cr_rev = np.array([idct(part) for part in q_Cr_rev])


    Y_rev = into_one_im(DCT_Y_rev, height, width)
    Cb_rev = into_one_im(DCT_Cb_rev, new_height, new_width)
    Cr_rev = into_one_im(DCT_Cr_rev, new_height, new_width)

    Cb_upsample = upsampling(Cb_rev, height_koef, width_koef)
    Cr_upsample = upsampling(Cr_rev, height_koef, width_koef)


    rgb_result = YCbCrtoRGB(Y_rev, Cb_upsample, Cr_upsample)

    img_decoded = Image.fromarray(rgb_result, 'RGB')
    img_decoded.save('img_decoded.jpg')

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(image)
    axs[1].imshow(img_decoded)

    axs[0].axis('off')
    axs[1].axis('off')

    axs[0].set_title('Before')
    axs[1].set_title('After')

    fig.patch.set_facecolor((255/255, 228/255, 225/255))
    plt.show()

# red_img()
main_function("puppy.png")
# main_function("solid_color_image.png")
# random_img()
# main_function("random_color_image.png")