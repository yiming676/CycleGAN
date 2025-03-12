import torch.nn as nn
import torch.nn.functional as F
import torch

class resBlock(nn.Module):
    def __init__(self,in_channel):
        super(resBlock,self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel ,3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self,x):
        return x+self.conv_block(x)

#生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        net = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # downsample
        in_channel = 64
        out_channel = in_channel * 2
        for _ in range(2):
            net += [
                nn.Conv2d(in_channel, out_channel, 3,stride=2,padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ]
            in_channel = out_channel
            out_channel = in_channel * 2

        for _ in range(9):
            net += [resBlock(in_channel)]

        #upsample
        out_channel = in_channel // 2
        for _ in range(2):
            net += [nn.ConvTranspose2d(in_channel,
                                       out_channel,
                                       3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)]
            in_channel = out_channel
            out_channel = in_channel // 2

        net += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,3,7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*net)

    def forward(self,x):
        return self.model(x)

#判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        model = [nn.Conv2d(3,64,4,stride=2,padding=1),
                 nn.LeakyReLU(0.2,inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                 nn.InstanceNorm2d(128),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                 nn.InstanceNorm2d(256),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, stride=2, padding=1),
                 nn.InstanceNorm2d(512),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512,1,4,padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        x = self.model(x)
        return F.avg_pool2d(x,x.size()[2:]).view(x.size()[0],-1)

if __name__ =='__main__':
    G = Generator()
    D = Discriminator()
    input_tensor = torch.ones((1,3,256,256),dtype = torch.float)
    out = G(input_tensor)
    print(out.size())

    out = D(input_tensor)
    print(out.size())

