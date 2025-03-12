# <center><font face="Arial" size="18" color="blue">CycleGAN</font></center>
***Generative Adversarial Networks*** (GANs) are a deep - learning model consisting of a **generator** and a **discriminator**. They **compete** to boost performance. The generator aims to **convert** images as realistically as possible, while the discriminator judges whether images are **real** or **generator - created**. CycleGAN is an **unsupervised - learning** GAN - based model. It handles image - to - image translation without **paired image data**. CycleGAN can change image styles and learn **mappings** between different image domains without paired data.

CycleGAN's core is **cycle - consistency loss**, which ensures that converted images, when translated back to the original domain, **retain** the input images' structural info. Unlike traditional GANs, CycleGAN uses **two generators** and **two discriminators** for image mapping. One generator creates mappings from the source to target domain, and the discriminator determines if images are from the target domain. A "reverse" generator and cycle - consistency loss ensure mapping **reliability**.

CycleGAN's **innovations** include unsupervised learning, cycle - consistency loss, and self - supervised learning. Unsupervised learning means no paired data is needed; CycleGAN learns source - to - target domain mappings with two generators and discriminators. Cycle - consistency loss ensures generated images can be converted back to the original, preserving structure. Self - supervised learning allows CycleGAN to train using an image's own structural info without direct labels.

CycleGAN has many uses, like art style transfer, season changes, object deformation, photo enhancement, and medical image processing. For example, it can turn photos into specific artist - style paintings, switch seasons in images, or convert between imaging modalities. CycleGAN achieves cross - domain translation of unpaired images through **adversarial training**, **cycle - consistency**, and **identity - mapping constraints**.

**Dataset Download and Visualization**: First, download the **apple2orange** dataset from ImageNet. It contains apple and orange images, all **scaled to 256×256 pixels**. There are 996 training apple images, 1020 training orange images, 266 test apple images, and 248 test orange images.

**Training and Testing**: During training, CycleGAN's generators and discriminators **adversarially train** to learn apple - to - orange translation. In testing, use a pre - trained or self - trained model to **translate test - set apples to oranges**.

**Output**: After successful testing, you'll see the **style - transferred images** and can directly view the results.

This is about CycleGAN from Lee Hongyi's neural - network course. If you use it, remember to **like Lee Hongyi's video**.
***
***生成对抗网络***（GANs）是一种深度学习模型，由 **生成器** 和 **判别器** 组成，它们相互竞争以提高性能。生成器的任务是尽可能真实地 **转换** 图像，而判别器的任务是 **判断** 图像是真实的还是由生成器生成的。CycleGAN是一种基于生成对抗网络（GAN）的**无监督学习**模型，旨在解决**没有成对图像数据**的图像到图像转换问题。CycleGAN不仅可以实现图像风格的转换，还能够在没有成对数据的情况下，学习到不同领域图像之间的映射关系。
CycleGAN的核心思想是通过引入**循环一致性损失**来确保生成的图像在转换回原始域时，能够保持与输入图像相同的结构信息。与传统的生成对抗网络不同，CycleGAN不需要成对的训练数据，它通过**两个**生成器和**两个**判别器来实现图像到图像的映射。生成器负责生成从源域到目标域的映射，判别器则用于判断图像是否来自目标域。为了确保映射的可靠性，CycleGAN还引入了“逆向”生成器，并通过循环一致性损失来确保图像的可逆性。
CycleGAN的创新点包括无监督学习、循环一致性损失和自监督学习。无监督学习意味着无需成对数据，CycleGAN通过两个生成器和两个判别器学习从源域到目标域的映射。循环一致性损失确保生成图像能够转换回原始图像，保持图像结构的完整性。自监督学习则是在没有直接标签的情况下，CycleGAN利用图像自身的结构信息进行训练。
CycleGAN在多个领域展现出了强大的应用潜力，包括艺术风格迁移、季节转换、物体变形、照片增强和医学图像处理。例如，它可以将普通照片转换成特定艺术家风格的画作，实现不同季节之间的图像转换，或在不同成像模态之间转换。CycleGAN通过**对抗训练**、**循环一致性**和**恒等映射的约束**，成功实现了无配对图像数据的跨域转换。


- 数据集下载与展示：首先，你需要下载**apple2orange**数据集，该数据集来源于ImageNet，包含了苹果和橘子的图像。图像被统一缩放为256×256像素大小，其中用于训练的苹果图片996张、橘子图片1020张，用于测试的苹果图片266张、橘子图片248张。
- 训练与测试过程：在训练过程中，CycleGAN会通过生成器和判别器的对抗训练来学习从苹果图像到橘子图像的转换。测试时，你可以使用预训练的模型或者自己训练的模型来转换测试集中的苹果图像到橘子风格。
- 输出结果：测试成功后，你将看到转换后的图像，你可以直观地看到风格迁移后的图像效果。

这是 李宏毅 老师的神经网络课程上讲解过的生成对抗网络--CycleGAN，大家使用的话记得给李宏毅老师的视频点赞haha
