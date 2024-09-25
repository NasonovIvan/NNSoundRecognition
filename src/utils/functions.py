import src.utils.imports as im


# mean data
def mean_data(data):
    mean = data[:].mean(axis=0)
    data -= mean
    std = data[:].std(axis=0)
    data /= std
    return data


# normalize images
def norm_images(data):
    return data[:] / 255.0


# plot spectrogram func
def PlotSpecgram(P, freqs, bins):
    Z = im.np.flipud(P)  # flip rows so that top goes to bottom, bottom to top, etc.
    xextent = 0, im.np.amax(bins)
    xmin, xmax = xextent
    extent = xmin, xmax, freqs[0], freqs[-1]

    img = im.pl.imshow(Z, extent=extent)
    im.pl.axis("auto")
    im.pl.xlim([0.0, bins[-1]])
    im.pl.ylim([0, 1000])


# Reads the frames from the audio clip and returns the uncompressed data
def ReadAIFF(file):
    s = im.aifc.open(file, "r")
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return im.np.fromstring(strSig, im.np.short).byteswap()


# creating small good spectrograms from wavs for article whales data
def create_images_data(path_train_audio, path_train_img, width=180, height=190):
    onlywavfiles = [
        f
        for f in im.listdir(path_train_audio)
        if im.isfile(im.join(path_train_audio, f))
    ]
    LENGTH = 256

    progress_counter_1 = 0
    progress_counter_2 = 10

    for file in onlywavfiles:
        progress_counter_1 += 1

        whale_sample_file = path_train_audio + file

        fs, x = im.read(whale_sample_file)
        f, t, Zxx_first = im.signal.stft(
            x,
            fs=fs,
            window=("hamming"),
            nperseg=LENGTH,
            noverlap=int(0.875 * LENGTH),
            nfft=LENGTH,
        )

        Zxx = im.np.log(im.np.abs(Zxx_first)) ** 2

        px = 1 / im.plt.rcParams["figure.dpi"]

        fig = im.plt.figure(
            figsize=(width * px, height * px), frameon=False
        )  # 180, 190 (139x146) for Kaggle dataset
        ax1 = im.plt.subplot()
        ax1.pcolormesh(t, f, Zxx, cmap="viridis")
        ax1.set_axis_off()
        im.plt.savefig(
            path_train_img + file[0:-4:1] + ".png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=100,
        )
        fig.clear()
        im.plt.close(fig)

        if (progress_counter_1 / len(onlywavfiles)) * 100 >= progress_counter_2:
            print((progress_counter_1 / len(onlywavfiles)) * 100, "% completed")
            progress_counter_2 += 10


# new function for creating images from audio files (more detailed)
def create_new_images(audio_folder, output_folder):
    """
    -- Input --
        - audio_folder - path to the dir with audio (.aiff)
        - output_folder - path to save images of spectrograms

    -- Output --
        - create spectrograms images in output folder
    """

    # Create a folder for saving the dataset if it doesn't exist
    im.os.makedirs(output_folder, exist_ok=True)

    # List of audio files we want to process
    file_list = im.os.listdir(audio_folder)

    n_fft = 200  # frame duration = n_fft /sr. If we want to get a 0.1-second frame duration at 2kHz our value must be 200.
    hop_length = 40  # 0.2*n_fft
    n_mels = 50  # Ajusted this value till the image had a good resolution
    fmax = 500  # Limiting the spectrogram to only 500 Hz frequency

    for filename in file_list:
        # Load the audio file
        audio_file = im.os.path.join(audio_folder, filename)
        samples, sr = im.sf.read(audio_file)

        # Create the spectrogram
        S = im.librosa.feature.melspectrogram(
            y=samples,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
        )
        S_db = im.librosa.power_to_db(S, ref=im.np.max)

        # Get the filename without extension
        output_filename = im.os.path.splitext(filename)[0] + ".png"

        # Save the spectrogram to a file with the corresponding name
        output_file = im.os.path.join(output_folder, output_filename)
        im.librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None
        )
        im.plt.axis("off")  # Turn off axis labels
        im.plt.savefig(output_file, bbox_inches="tight", pad_inches=0, transparent=True)
        im.plt.close()


# create Kaggle dataset
def CreateKaggleDataset(df, path_train_img, slice_name=5):
    """
    CreateKaggleDataset function

    --Input--
    df (DataFrame): train.csv file from Kaggle
    train_index (int): how many samples should be in training set
    val_index (int): how many samples should be in validation set
    path_train_img (str): path to the images (clear or noised)
    slice_name (int): number, which mean image file name in path (5 for kaggle and 4 for article)

    --Output--
    x_data (array)
    y_data (array)
    """
    x_data = []
    y_data = []

    for i in range(len(df["clip_name"])):
        FILENAME = (
            path_train_img + df["clip_name"][i][:-slice_name] + ".png"
        )  # slice_name=5 for kaggle and slice_name=4 for article
        rgba_image = im.Image.open(FILENAME)
        img = rgba_image.convert("RGB")
        img_arr = im.np.asarray(img)
        rgba_image.close()
        # print(FILENAME)

        x_data.append(img_arr)
        y_data.append(df["label"][i])

    return im.np.array(x_data), im.np.array(y_data)


# Defining class to add random noise (Ornstein-Uhlenbeck_process) in actions to increase Exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=0.1, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev
            * im.np.sqrt(self.dt)
            * im.np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = im.np.zeros_like(self.mean)


# noise object for augmentation
std_dev = 0.01
noise_object = OUActionNoise(
    mean=im.np.zeros(1), std_deviation=float(std_dev) * im.np.ones(1)
)


# creating augmentation Kaggle dataset for Xception Network
def CreateKaggleDatasetNew(csv_file, image_folder, target_size=(255, 255)):
    """
    -- Input --
    - csv_file - Path to the CSV file with labels
    - image_folder - Path to the folder containing spectrogram images

    """
    # Load labels from the CSV file
    labels_df = im.pd.read_csv(csv_file)

    # Create a dictionary of labels for each file
    labels_dict = {}
    for _, row in labels_df.iterrows():
        filename = row["clip_name"]
        label = row["label"]
        labels_dict[filename] = str(label)

    # Get a list of image files
    image_files = im.os.listdir(image_folder)

    # Create a list of (image file path, label) pairs
    image_label_pairs = [
        (
            im.os.path.join(image_folder, filename),
            labels_dict.get(filename[:-4] + ".aiff"),
        )
        for filename in image_files
    ]

    # Create a list of (image file path, label) pairs with label equal to 1 (only whales)
    image_label_pairs_whales = [
        (
            im.os.path.join(image_folder, filename),
            labels_dict.get(filename[:-4] + ".aiff"),
        )
        for filename in image_files
        if labels_dict.get(filename[:-4] + ".aiff") == "1"
    ]

    # Data augmentation function
    def augment_image(image, label):
        # Random color augmentation
        image = im.tf.image.random_hue(image, 0.08)
        image = im.tf.image.random_saturation(image, 0.6, 1.6)
        image = im.tf.image.random_brightness(image, 0.05)

        return image, label

    # Load and preprocess images
    def load_preprocess_images(image_label_pairs):
        images = []
        for image_path, _ in image_label_pairs:
            image = im.tf.io.read_file(image_path)
            image = im.tf.image.decode_png(image, channels=3)
            image = im.tf.image.convert_image_dtype(image, im.tf.float32)
            image = im.tf.image.resize(image, target_size)  # Resize the image
            images.append(image)
        return images

    # Get labels from pairs (image, label)
    def get_labels(image_label_pairs):
        labels = [int(label) for _, label in image_label_pairs]
        return labels

    images_original = load_preprocess_images(image_label_pairs)
    labels_original = get_labels(image_label_pairs)

    # print("orig", images_original)

    images_augmented = load_preprocess_images(image_label_pairs_whales)
    labels_augmented = get_labels(image_label_pairs_whales)

    original_dataset = im.tf.data.Dataset.from_tensor_slices(
        (images_original, labels_original)
    )
    augmented_dataset = im.tf.data.Dataset.from_tensor_slices(
        (images_augmented, labels_augmented)
    )
    augmented_dataset = augmented_dataset.map(augment_image)  # Apply data augmentation

    # Combine the original and augmented datasets
    combined_dataset = original_dataset.concatenate(augmented_dataset)

    return combined_dataset


# Function for creating spectrograms' dataset from the article data
def create_article_dataset(csv_file, image_folder, image_size=(255, 255)):
    # Load the CSV file with labels
    df = im.pd.read_csv(csv_file, sep=";")

    # Function to load and preprocess images
    def load_and_preprocess_image(image_path, label):
        # Load the image
        image = im.tf.io.read_file(image_path)
        image = im.tf.image.decode_image(image, channels=3, expand_animations=False)
        # image = tf.image.decode_image(image, channels=3)
        # Resize the image to the specified size
        image = im.tf.image.resize(image, image_size)
        # Normalize pixel values to the range [0, 1]
        image = im.tf.cast(image, im.tf.float32) / 255.0
        return image, label

    # Create a list of image paths and labels
    image_paths = [
        im.os.path.join(image_folder, filename[:-4] + ".png")
        for filename in df["clip_name"]
        if filename[0] != "A"
    ]
    labels = [
        int(df["label"][i])
        for i in range(len(df["label"]))
        if df["clip_name"][i][0] != "A"
    ]

    # Create a dataset using tf.data.Dataset
    dataset = im.tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Load and preprocess images
    dataset = dataset.map(load_and_preprocess_image)

    return dataset, im.np.array(labels)


# function for adding noise to data
def add_noise_or_change(example):
    noise = noise_object()
    # augmented_example = example + im.np.random.normal(0, 0.008, size=example.shape)
    augmented_example = example + noise
    return augmented_example


# function for data augmentation by adding noise to data and
# increase the number of samples with the label 1
def augment_data(x_data, y_data, augmentation_factor=2):
    x_augmented = []
    y_augmented = []

    x_augmented.extend(x_data)
    y_augmented.extend(y_data)

    # Getting indexes of examples with label 1
    idx_label_1 = im.np.where(y_data == 1)[0]

    # Create additional examples labelled 1
    for _ in range(augmentation_factor):
        # We select a random index from the examples labelled 1
        random_idx = im.np.random.choice(idx_label_1)
        example_label_1 = x_data[random_idx]

        # Add random noise or changes for example with mark 1
        augmented_example = add_noise_or_change(example_label_1)

        x_augmented.append(augmented_example)
        y_augmented.append(1)

    x_augmented = im.np.array(x_augmented)
    y_augmented = im.np.array(y_augmented)

    return x_augmented, y_augmented


# for plotting loss and accuracy history training
def PlotLossAcc(
    TrainData,
    ValData,
    Epochs,
    TrainLabel,
    ValLabel,
    yLabel,
    title,
    ColTrain,
    ColVal,
    filename,
    lr=False,
):
    pdf = im.PdfPages(filename)
    fig, ax = im.plt.subplots(figsize=(8, 6))
    ax.plot(Epochs, TrainData, color=ColTrain, label=TrainLabel)
    if lr == False:
        ax.plot(Epochs, ValData, color=ColVal, label=ValLabel)
    if title != False:
        im.plt.title(title)
    ax.set_ylabel(yLabel, fontsize=16)
    ax.set_xlabel(r"$Epochs$", fontsize=16)
    im.plt.tick_params(axis="both", which="major", labelsize=14)
    if lr == False:
        im.plt.legend(fontsize=14)
    im.plt.grid()
    pdf.savefig(fig)
    pdf.close()
    im.plt.show()


# plot data distribution
def PlotDataDistribution(
    samples, counts, bar_labels, bar_colors, filename, legend=True
):
    pdf = im.PdfPages(filename)

    fig, ax = im.plt.subplots()

    bar_container = ax.bar(samples, counts, label=bar_labels, color=bar_colors)

    for bar in bar_container:
        height = bar.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set(ylim=(0, 25000))

    # ax.legend(title='Labels')
    if legend == True:
        ax.legend()
    pdf.savefig(fig)
    pdf.close()
    im.plt.show()


# function for making h/c train and test from wavs
def compute_spectral_info(spectrum):
    N = len(spectrum)

    H_p = 0
    H_q = 0
    Complexity_sq = 0
    Complexity_jen = 0
    Complexity_abs = 0

    yf = spectrum

    Sum = sum(yf)
    square_sum = 0
    abs_sum = 0

    p_is = []
    for s in yf:

        p_i = s / Sum
        if p_i > 0:
            p_is.append(p_i)
            H_p += -p_i * im.np.log2(p_i)

    Nfft = len(p_is)
    q_i = 1.0 / Nfft  # Noise spectrum

    for k in range(Nfft):
        square_sum += (p_is[k] - q_i) ** 2
        abs_sum += im.np.abs(p_is[k] - q_i)

    Disequilibrium_sq = square_sum

    H_p /= im.np.log2(Nfft)

    Jensen = im.jensenshannon(p_is, [q_i for j in range(Nfft)])

    Q0 = -2.0 / (
        (Nfft + 1) * im.np.log2(Nfft + 1) / Nfft
        - 2 * im.np.log2(2 * Nfft)
        + im.np.log2(Nfft)
    )

    Disequilibrium_jen = (Jensen**2) * Q0

    Complexity_sq = H_p * square_sum

    Complexity_jen = (
        H_p * Disequilibrium_jen
    )  # H_p*square_sum #H_p*(Jensen**2)*Q0 ##H_p*square_sum ##np.exp(H_p)*square_sum # H_p*(0.5*square_sum*len(p_is) - 1.0*third_sum*(len(p_is)**2)/6)

    Complexity_abs += H_p * (abs_sum**2) / 4

    return H_p, Complexity_sq, Complexity_jen, Complexity_abs


# create Complexity-Entropy datasets, based on the compute_spectral_info function
def create_HC_dataset_wavs(df, path_train, Noised=False, binary=False):
    x_data = []
    y_data = []

    progress_counter_1 = 0
    progress_counter_2 = 10

    for i in im.tqdm.tqdm(range(len(df["clip_name"]))):
        if Noised:
            FILENAME = path_train + "noised_002_" + df["clip_name"][i][:-5] + ".wav"
        else:
            FILENAME = path_train + df["clip_name"][i][:-5] + ".wav"
            # FILENAME = path_train + df["clip_name"][i]

        H_s = []
        C_sqs = []
        C_jsds = []
        C_tvs = []

        WINDOW_FFT = 256
        WINDOW_HOP = 32
        y_librosa, sr_librosa = im.librosa.load(FILENAME, sr=None)
        y_librosa -= im.np.mean(y_librosa)
        M = im.librosa.feature.melspectrogram(
            y=y_librosa,
            sr=sr_librosa,
            htk=True,
            hop_length=WINDOW_HOP,
            n_fft=WINDOW_FFT,
            n_mels=64,
            fmin=50,
            fmax=300,
        )
        M_db = im.librosa.power_to_db(M, ref=im.np.min) ** 2

        # for j in range(10, 85):
        for j in range(126):
            local_spect = M_db[:, j]

            H, C_sq, C_jsd, C_tv = compute_spectral_info(local_spect)
            H_s.append(H)
            C_sqs.append(C_sq)
            C_jsds.append(C_jsd)
            C_tvs.append(C_tv)

        hc_plane = (H_s, C_sqs, C_jsds, C_tvs)

        x_data.append(hc_plane)
        y_data.append(df["label"][i])

        # if (progress_counter_1 / len(df["clip_name"])) * 100 >= progress_counter_2:
        #     print((progress_counter_1 / len(df["clip_name"])) * 100, '% completed')
        #     progress_counter_2 += 10

    return im.np.array(x_data), im.np.array(y_data)


# compute Complexity-Entropy data from wav-file with ordpy
def compute_hc_from_ordpy(FILENAME):

    audio_signal, RATE = im.librosa.load(FILENAME, sr=None)
    audio_signal -= im.np.mean(audio_signal)

    # audio_signal -= int(np.mean(audio_signal))

    window_size = 256

    H_ps = []
    Complexitys = []
    Fishers = []

    Hs_my = []
    Complexities_sq_my = []
    Complexities_jen_my = []
    Complexities_tv_my = []

    start_point = 0
    end_point = 0

    i = 0

    while start_point < len(audio_signal):

        end_point = start_point + window_size

        frame_audio = audio_signal[start_point:end_point:1]

        start_point = start_point + 32

        HC = im.ordpy.complexity_entropy(frame_audio, dx=4)

        (patterns, probs) = im.ordpy.ordinal_distribution(frame_audio, dx=4)

        H_my = 0
        Disequlibrium_sq_my = 0
        Disequlibrium_jen_my = 0
        Disequlibrium_tv_my = 0
        N = len(probs)
        q_i = 1.0 / N

        for prob in probs:
            H_my += -prob * im.np.log2(prob)
            Disequlibrium_sq_my += (prob - q_i) ** 2
            Disequlibrium_tv_my += im.np.abs(prob - q_i)

        Disequlibrium_jen_my = im.jensenshannon(probs, [q_i for j in range(N)])

        H_my /= im.np.log2(N)

        Hs_my.append(H_my)
        Complexities_sq_my.append(H_my * Disequlibrium_sq_my)
        Complexities_jen_my.append(H_my * (Disequlibrium_jen_my**2))
        Complexities_tv_my.append(H_my * (Disequlibrium_tv_my**2))

        # HF = ordpy.fisher_shannon(frame_audio, dx = 3)

        H_ps.append(HC[0])
        Complexitys.append(HC[1])
        # Fishers.append(HF[1])

        i += 1

    H_ps = H_ps[10:85:1]
    Complexitys = Complexitys[10:85:1]
    Complexities_sq_my = Complexities_sq_my[10:85:1]
    Complexities_jen_my = Complexities_jen_my[10:85:1]
    Complexities_tv_my = Complexities_tv_my[10:85:1]
    return (
        H_ps,
        Complexitys,
        Complexities_sq_my,
        Complexities_jen_my,
        Complexities_tv_my,
    )


# creating train, validation, test datasets with ordpy function
def create_hc_dataset_ordpy(df, path_train, train_index, val_index):
    labels = []
    for label in df["label"]:
        labels.append(int(label))
    labels = im.np.array(labels)

    HC_list = []

    progress_counter_1 = 0
    progress_counter_2 = 10

    for i in range(len(df["clip_name"])):
        HC_tmp = []
        progress_counter_1 += 1
        # FILENAME = path_train + df["clip_name"][i]
        FILENAME = path_train + df["clip_name"][i][:-5] + ".wav"
        (
            H_ps,
            Complexitys,
            Complexities_sq_my,
            Complexities_jen_my,
            Complexities_tv_my,
        ) = compute_hc_from_ordpy(FILENAME)
        HC_tmp.append(H_ps)
        HC_tmp.append(Complexitys)
        HC_tmp.append(Complexities_sq_my)
        HC_tmp.append(Complexities_jen_my)
        HC_tmp.append(Complexities_tv_my)
        HC_list.append(HC_tmp)

        if (progress_counter_1 / len(df["clip_name"])) * 100 >= progress_counter_2:
            print((progress_counter_1 / len(df["clip_name"])) * 100, "% completed")
            progress_counter_2 += 10

    HC_array = im.np.array(HC_list)
    hc_train = HC_array[:train_index]
    hc_y_train = labels[:train_index]
    hc_val = HC_array[train_index:val_index]
    hc_y_val = labels[train_index:val_index]
    hc_test = HC_array[val_index:]
    hc_y_test = labels[val_index:]

    return hc_train, hc_y_train, hc_val, hc_y_val, hc_test, hc_y_test


# ==== evaluate functions recall, precision and f1-score for neural network ====
def recall(y_true, y_pred):
    y_true = im.K.ones_like(y_true)
    true_positives = im.K.sum(im.K.round(im.K.clip(y_true * y_pred, 0, 1)))
    all_positives = im.K.sum(im.K.round(im.K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + im.K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = im.K.ones_like(y_true)
    true_positives = im.K.sum(im.K.round(im.K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = im.K.sum(im.K.round(im.K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + im.K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + im.K.epsilon()))


# Defining binary focal loss function
def binary_focal_loss(gamma=2.0, alpha=0.25):
    def tf_binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = im.tf.where(im.tf.equal(y_true, 1), y_pred, im.tf.ones_like(y_pred))
        pt_0 = im.tf.where(im.tf.equal(y_true, 0), y_pred, im.tf.zeros_like(y_pred))
        epsilon = im.K.epsilon()
        pt_1 = im.K.clip(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = im.K.clip(pt_0, epsilon, 1.0 - epsilon)
        return -im.K.sum(
            alpha * im.K.pow(1.0 - pt_1, gamma) * im.K.log(pt_1)
        ) - im.K.sum((1 - alpha) * im.K.pow(pt_0, gamma) * im.K.log(1.0 - pt_0))

    return tf_binary_focal_loss_fixed


# plot article dataset examples for Xception Network
def PlotArticleExample(article_test, num):
    for image, label in article_test.take(num):
        # Convert the TensorFlow tensor to a NumPy array
        image = image.numpy()
        print(image.shape)

        # Plot the image
        im.plt.imshow(image)
        im.plt.title(f"Label: {label}")
        im.plt.axis("off")  # Turn off axis labels
        im.plt.show()
