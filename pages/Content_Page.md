## Document
	- ### [[Research Plan]]
-
- # Neuron-based/AI-related Models
	- ### [[Digital Twin Brain]]
- # Population-based Models
	- ## Brain Network Model
		- ### [[References of BNM]]
		- ### Main Researchers
			- ### [[Viktor Jirsa]]
			- ### [[Gustavo Deco]]
			- ### [[Wang Xiao-jing]]
			- ### [[Joana Cabral]]
			- ### [[Pedro A. Valdes-Sosa]]
			- ### [[John D. Murray]]
			- ### [[Thomas Yeo]]
			- ### [[Danielle Bassett]]
			- ### [[Olaf Sporns]]
		- ### Topics
			- ### [[Heterogenous Modeling]]
			- ### [[Modeling Neuromodulation]]
			- ### [[Spatial-temporal Hierarchy]]
			- ### [[Comparison And Validation of BNM]]
			- ### [[Model Stimulation And Control]]
			- ### [[Model for Other Modal]]
			- ### Application to Diseases
				- ### [[Schizophrenia]]
				- ### [[Epilepsy]]
				- ### [[Alzheimer's Disease]]
		- ### Neural Mass Models
			- ### Biophysical Model
				- ### [[Wong-Wang Model]]
				- ### [[Wilson-Cowan Model]]
			- ### Phenomenal Model
				- ### [[Kuramoto Model]]
		- ### Novel Ideas
			-
		- ### Questions
			- Target：围绕模型的个体化、泛化能力、过拟合问题之类的做一些讨论
			- **如何理解神经群体模型所模拟的群体活动的功能意义，及其与宏观、微观生理过程、生理指标的关系**
				- 如何解释使用群体平均活动（平均发放率等在结构连接上平均地传递）模拟宏观功能特征的合理性及其生理意义，为什么可以这么做，这种建模方式描述了怎样的生理过程，可以回答什么生物问题
				- 如何构建神经群体模型的动力学特性（参数）与微观生理特性的关联，如何在真实数据中检验这种关联
			- **如何评估模型的表征能力和辨别能力**
				- 模型的动力学性态的变化范围是否足够包含真实的结构-功能关系（健康、疾病、任务态、个体差异）
				- 模型的动力学参数是否足够敏感以区分不同的结构-功能关系（健康、疾病、任务态、个体差异）
			- 有没有可能利用 **AI** 学习一个 **动力学性态+物理连接 预测 功能连接** 的模型，再利用这个模型去估计真实连接的动力学性态？
				- 或者换一个方向，学习 **物理连接+功能连接 预测 动力学性态？**
				- 这涉及到两个问题：
					- 如何设置一个可以表征尽可能多性态的二维节点模型（稳定点|位置+收敛速度、极限环|频率+幅值、临界态|？）
					  logseq.order-list-type:: number
					- 如何判定模拟数据的采样是充分的，模拟数据中模拟的结构-功能对应关系可以涵盖真实数据的范围（隐空间似乎有用）
					  logseq.order-list-type:: number
			- 在目前的尝试中，仅有最粗糙的结构模板DK68和AAL90能够取得较好的全脑FC模拟效果，**如何将模型推广到更细致的脑区模版，甚至推广到 voxel/vertex 的介观尺度**
	-
	- ## Neural Field Model
		- ### [[References of NFM]]
		- ### Main Researchers
			- ### [[Peter A. Robinson]]
			- ### [[Micheal Breakspear]]
		- ### Topics
			- ### [[Cortical Waves]]
			- ### Application to Diseases
		- ### Novel Ideas
			-
		- ### Questions
			-
-
- ## Works Related to Neural Dynamics
	- ### Reviews
		- [[On the nature and use of models in network neuroscience - 2018 - 193]] - 脑网络中的模型
		  collapsed:: true
			- 网络理论为研究相互关联的大脑机制之间的关系及其与行为的相关性提供了一个直观而吸引人的框架。网络模型的多样性可能导致混乱，使评估模型有效性和效力的工作复杂化，并妨碍跨学科合作。在这篇综述中，我们检查了网络神经科学领域，重点放在组织可以帮助克服这些挑战的原理。首先，我们描述了构建网络模型的基本目标。其次，我们回顾了最常见的网络模型形式。网络模型可以从三个维度加以描述：从数据表征到第一性原理理论；从生物物理现实主义|biophysical realism到功能现象学|functional phenomenology；从基本描述|elementary description到粗粒度近似|coarse-grained approximation。第三，我们利用生物学、哲学和其他学科为这些模型建立验证原则。最后，我们将讨论沟通不同模型类型的机会，并为未来的追求指出令人兴奋的前沿。
		- [[The physics of brain network structure, function and control - 2019 - 121]] - 脑网络经典综述
		  collapsed:: true
			- 大脑的特点是结构连接的异质模式支持无与伦比的认知能力和广泛的行为。新的非侵入性成像技术现在可以对这些模式进行全面映射。然而，了解大脑的结构线路如何支持认知过程，以及对个性化心理健康治疗有重大影响，仍然是一个根本性的挑战。在这里，我们回顾了最近为迎接这一挑战所做的努力，利用物理直觉、模型和理论，跨越统计力学、信息论、动力系统和控制领域。我们首先描述了在空间嵌入和能量最小化的约束下在结构布线中实例化的脑网络架构的组织原则。然后，我们调查了脑网络功能模型，这些模型规定了神经活动如何沿着结构连接传播。最后，我们讨论了脑网络控制的微扰实验和模型；这些利用沿着结构连接的信号传输物理学来推断支持目标导向行为的内在控制过程，并为神经和精神疾病的基于刺激的疗法提供信息。自始至终，我们都强调了邀请先驱物理学家进行创造性努力的开放性问题。
		- [[Linking Structure and Function in Macroscale Brain Networks - 2020 - 407]] - **结构功能关系**综述
		  collapsed:: true
			- 网络神经科学的出现使研究人员能够量化神经元网络的组织特征和皮层功能谱之间的联系。目前的模型表明，结构与功能之间存在显著的相关性，但由于功能反映了结构网络中复杂的多突触相互作用，因此这种相关性并不完美。功能不能直接从结构上估计，而必须通过高阶相互作用模型来推断。统计、通讯和生物物理模型已经被用来将大脑结构转化为大脑功能。结构-功能耦合具有区域异质性，遵循分子、细胞结构和功能层次。
		- [[Macroscopic gradients of synaptic excitation and inhibition in the neocortex - 2020]] - 兴奋抑制梯度 综述
		  collapsed:: true
			- 随着连接组学、转录组学和神经生理学技术的进步，大脑神经回路的神经科学即将起飞。一个主要的挑战是理解哺乳动物新皮层的分布区域是如何服务于大量不同的功能的，这些区域由规范的局部电路重复组成。大脑皮层的各个区域不仅在输入-输出模式上彼此不同，而且在生物学特性上也各不相同。最近的实验和理论研究表明，这种变化不是随机的异质性，而是突触兴奋和抑制在整个大脑皮层表现出系统的宏观梯度，在精神疾病中是不正常的。在非线性神经动力学系统中，**沿着这些梯度的定量差异可以通过数学上描述为分岔的现象导致定性的新行为**。宏观梯度和分叉的结合，与生物进化，发育和可塑性相结合，提供了皮层区域之间功能多样性的生成机制，作为大规模皮层组织的一般原则。
		- [[Reconstructing computational system dynamics from neural data with recurrent neural networks - 2023]] - 神经动力学反问题
		  collapsed:: true
			- 神经科学中的计算模型通常采用微分方程组的形式。此类系统的行为是动力系统理论的主题。动力系统理论为分析神经生物学过程提供了强大的数学工具箱，几十年来一直是计算神经科学的支柱。最近，循环神经网络（RNN）已成为一种流行的机器学习工具，用于通过模拟底层微分方程系统来研究神经和行为过程的非线性动力学。 RNN 经常接受与动物受试者类似的行为任务训练，以生成有关底层计算机制的假设。相比之下，RNN 还可以根据测量的生理和行为数据进行训练，从而直接继承它们的时间和几何特性。通过这种方式，它们成为实验探测系统的正式替代品，可以进一步分析、扰动和模拟。这种强大的方法称为动态系统重建。在本视角中，我们重点关注人工智能和机器学习这个令人兴奋且迅速发展的领域的最新趋势，而该领域在神经科学中可能不太为人所知。我们讨论基于 RNN 的动态系统重建的形式先决条件、不同的模型架构和训练方法、评估和验证模型性能的方法、如何在神经科学背景下解释训练模型以及当前的挑战。
		- [[Towards a biologically annotated brain connectome - 2023]] - 由节点特征标记的结构连接
		  collapsed:: true
			- 大脑是由交错的神经回路组成的网络。在现代连接组学中，大脑连接通常被编码为一个由节点和边缘组成的网络，抽象出当地神经元群体丰富的生物学细节。然而，网络节点的生物学注释ーー例如基因表达、大脑皮质细胞结构学、神经递质受体或内在动力学ーー可以很容易地测量并覆盖在网络模型上。在这里，我们回顾如何连接体可以表示和分析作为注释网络。带注释的连接体使我们能够重新概念化网络的结构特征，并将大脑区域的连接模式与其基础生物学联系起来。新出现的工作表明，注释连接体有助于建立更加真实的大脑网络形成模型，神经动力学和疾病传播。最后，注释可以用来推断全新的区域间关系，并构建新的网络类型，以补充现有的连接体表示。总之，带有生物学注释的连接体提供了一种令人信服的方法来研究与当地生物特征相一致的神经连接。
		- [Theoretical foundations of studying criticality in the brain | Network Neuroscience | MIT Press](https://direct.mit.edu/netn/article/6/4/1148/112392/Theoretical-foundations-of-studying-criticality-in) - 大脑中的临界性
		  collapsed:: true
			- [综述 | 从第一性原理出发，探索类脑智能研究的星辰大海 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzU5ODg0MTAwMw==&mid=2247534407&idx=2&sn=ed33b3bcf36e17aaaec9888f90b2c027&chksm=febc0483c9cb8d959cb311fe79ee4ee29763c371e4acc5d44f734365fc5dd4b25e9767459834&mpshare=1&scene=2&srcid=0118Bn6PdArNh0UplKbIO1bf&sharer_shareinfo=ec8bc4ab5eb695f0464c62f3a353c87f&sharer_shareinfo_first=ec8bc4ab5eb695f0464c62f3a353c87f#rd)
		- [[The coming decade of digital brain research: A vision for neuroscience at the intersection of technology and computing - 2024]] - **数字大脑研究的未来十年：技术与计算交叉点的神经科学愿景**
		  collapsed:: true
			- 近年来，在方法论的重大进步以及从分子到整个大脑的多个尺度的数字化数据集成和建模的推动下，大脑研究无疑进入了一个新纪元。神经科学与技术和计算的交叉领域正在取得重大进展。这种新的大脑科学结合了高质量的研究、跨多个尺度的数据集成、多学科大规模协作的新文化以及转化为应用。正如欧洲人脑计划 (HBP) 所倡导的那样，系统性方法对于应对未来十年紧迫的医疗和技术挑战至关重要。本文的目的是：为未来十年的数字大脑研究制定一个概念，与广大研究界讨论这个新概念，确定共识点，并从中得出科学的共同目标；为 EBRAINS 当前和未来的发展提供科学框架，EBRAINS 是 HBP 工作产生的研究基础设施；向利益相关者、资助组织和研究机构通报未来数字大脑研究并让其参与其中；识别并解决人工智能综合大脑模型的变革潜力，包括机器学习和深度学习；概述了一种协作方法，将道德和社会机遇与挑战的反思、对话和社会参与结合起来，作为未来神经科学研究的一部分。
		- [Big data and the industrialization of neuroscience: A safe roadmap for understanding the brain? | Science](https://www.science.org/doi/abs/10.1126/science.aan8866#bibliography)
	- ### 理解大脑动力学
		- [[Emergent complex neural dynamics - 2010 - 681]] - 二阶相变临界点附近的动力系统中的涌现复杂现象作为大脑功能的产生机制
		  collapsed:: true
			- [Emergent complex neural dynamics | Nature Physics](https://www.nature.com/articles/nphys1803)
			- 大脑中丰富的时空活动模式是适应行为的基础。理解大脑的百亿神经元和百万亿突触如何以灵活的方式产生如此多种皮质结构的机制仍然是神经科学中的一个基本问题。一个可能的解决方案是涉及到紧邻二阶相变临界点的动力系统中显现的涌现复杂现象的普遍机制。我们回顾了最近理论和实证结果，支持大脑天然处于临界性附近这一观点，以及这对更好理解大脑的意义。
		- [[Criticality in the brain: A synthesis of neurobiology, models and cognition - 2017]] - 神经系统的临界性的证据
		  collapsed:: true
			- [Criticality in the brain: A synthesis of neurobiology, models and cognition - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0301008216301630)
			- 认知功能需要在许多尺度上协调神经活动，从神经元和电路到大规模网络。因此，一个针对任何单一尺度的解释框架不太可能产生一个关于大脑活动和认知功能的综合理论。神经科学的建模和分析方法应旨在适应多尺度现象。现在新兴的研究表明，大脑中的多尺度过程源于在自然界中非常广泛发生的所谓临界现象。临界性产生于介于有序与无序之间的复杂系统中，其特点是**时间和空间上无标度的波动**。我们回顾了临界性的核心性质、支持其在神经系统中的作用的证据及其在大脑健康和疾病中的解释潜力。
		- [[Is temporo-spatial dynamics the “common currency” of brain and mind? In Quest of “Spatiotemporal Neuroscience” - 2020 - 114]] - 时空动力学沟通神经活动与心理表征
		- [[How critical is brain criticality? - 2022]] - 大脑临界性综述
		- [[When Causality meets Inference: complexity in neuroscience - 2022]]
		- [[The free energy principle made simpler but not too simple - 2023]] - 最小自由能原理
	- ### 理解大脑功能
		- [[* What we can do and what we cannot do with fMRI - 2008 - 2209]] - 兴奋抑制网络与fMRI的生理解释
		  collapsed:: true
			- * What we can do and what we cannot do with fMRI - 2008 - 2209
			- [Regional variation in neurovascular coupling and why we still lack a Rosetta Stone | Philosophical Transactions of the Royal Society B: Biological Sciences (royalsocietypublishing.org)](https://royalsocietypublishing.org/doi/full/10.1098/rstb.2019.0634)
			- [Non-Neural Factors Influencing BOLD Response Magnitudes within Individual Subjects | Journal of Neuroscience (jneurosci.org)](https://www.jneurosci.org/content/42/38/7256.abstract)
			- [Causal mapping of human brain function | Nature Reviews Neuroscience](https://www.nature.com/articles/s41583-022-00583-8)
			- [Challenges and future directions for representations of functional brain organization | Nature Neuroscience](https://www.nature.com/articles/s41593-020-00726-z)
			- [Graph Neural Networks in Network Neuroscience | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9903566)
			- 功能磁共振成像（fMRI）目前是认知神经科学中神经影像学的支柱。扫描仪技术、图像采集协议、实验设计和分析方法的进步有望推动功能磁共振成像从单纯的制图到真正的大脑组织研究。然而，关于功能磁共振成像数据解释的基本问题比比皆是，因为得出的结论往往忽略了该方法的实际局限性。在这里，我概述了功能磁共振成像的现状，并利用神经影像和生理数据来介绍当前对血流动力学信号的理解及其对神经影像数据解释的限制。
		- [[Noise in the nervous system - 2008 - 1825]] - 神经系统中的噪声
		  collapsed:: true
			- 噪声——信号的随机干扰——给信息处理带来了一个基本问题，并影响神经系统功能的各个方面。然而，神经系统中噪声的性质、数量和影响直到最近才以定量的方式得到解决。实验和计算方法表明，多个噪声源会导致细胞和行为试验之间的变异性。我们从分子水平到行为水平回顾了神经系统中的噪音来源，并展示了噪音如何影响试验间的变异性。我们重点介绍噪声如何影响神经元网络以及神经系统用于对抗噪声有害影响的原理，并简要讨论噪声的潜在好处。
		- [[The dark matter of the brain - 2019]] - 大脑存在大量沉默神经元
		  collapsed:: true
			- [How silent is the brain: is there a “dark matter” problem in neuroscience? | SpringerLink](https://link.springer.com/article/10.1007/s00359-006-0117-6)
			- 大脑能量消耗的大部分用于维持神经元和神经回路的永久效能。然而，对麻醉和行为动物进行的长期电生理和神经影像学研究表明，完整大脑中的绝大多数神经细胞不会激发动作电位，即永久性地保持沉默。在这里，我回顾了新出现的数据，表明哺乳动物神经系统中的神经细胞大量冗余，以高能量代价维持在抑制状态。在进化过程中获得的这些休眠神经元和电路的集合逃避了日常的功能任务，因此不受自然选择的影响。然而，在穿透性压力和疾病下，它们偶尔会在活跃状态下转换，导致各种神经精神症状和行为异常。越来越多的证据表明，沉默神经元的广泛存在，需要认真修订大脑的功能模型，并需要不可预见的恢复和可塑性储备。
		- [[Human cognition involves the dynamic integration of neural activity and neuromodulatory systems - 2019 - 212]] —— **神经活动的低维流形**
		  collapsed:: true
			- 人脑将不同的认知过程整合成一个连贯的整体，并随着环境需求的变化而流畅地变化。尽管最近取得了进展，但对这种动态系统级整合的神经生物学机制仍然知之甚少。在这里，我们研究了一系列认知任务中全系统神经活动的空间、动态和分子特征。我们发现神经元活动汇聚到一个低维流形上，有利于执行不同的任务状态。该吸引子空间内的流动与可分离的认知功能、网络级拓扑的独特模式以及流体智力的个体差异相关。低维神经认知架构的轴与神经调节受体密度的区域差异一致，而神经调节受体密度的区域差异又与从结构连接组估计的网络可控性的不同特征相关。这些结果通过强调神经活动、神经调节系统和认知功能之间的相互作用，增进了我们对功能性大脑组织的理解。
		- [[Why context matters? Divisive normalization and canonical microcircuits in psychiatric disorders - 2020]] - 上下文依赖性的计算机制
		  collapsed:: true
			- 细胞、区域和行为水平上的神经活动显示出上下文依赖性。在这里，我们建议用除法归一化（DN）来处理输入-输出关系，包括（i）求和/平均输入和（ii）根据输入阶段归一化输出，作为一种计算机制，作为上下文依赖性的基础。输入求和和输出归一化由规范微电路 （CM） 中的输入输出关系介导。DN/CM 在精神分裂症或抑郁症等精神疾病中发生改变，其各种症状可通过异常的环境依赖性来表征。
		- [[Brain Activity Is Not Only For Thinking  - 2021 - 14]] - BOLD可能与固有的生理过程而非认知活动更相关
		  collapsed:: true
			- 人脑是一个复杂的器官，具有多个竞争性使命。它必须感知和解释世界，吸收新信息，并在整个寿命中保持其功能完整性。神经活动与所有这些过程相关联。自发的BOLD信号被引用为代表与所有这些过程相关联的神经活动。然而，它们在这些过程中的确切作用仍然备受争议。在这里，我们回顾学习机理论、突触可塑性和稳态的分子机制，以及最近的实验证据，以暗示自发的BOLD活动可能与离线可塑性和稳态过程更密切相关，而不是在线认知内容的波动。
		- [[* Neurodevelopment of the association cortices: Patterns, mechanisms, and implications for psychopathology - 2021 - 243]] - 沿S-A轴的皮层层次发育模式
		  collapsed:: true
			- [Neurodevelopment of the association cortices: Patterns, mechanisms, and implications for psychopathology - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0896627321004578)
			- 人类大脑经历了一个跨越数十年的长期皮层发育期。在儿童和青少年期间，皮层发育从具有感觉和运动功能的低阶，初级和单模式皮层进展到高阶，跨模式联合皮层，服务于执行，社会情绪和心理功能。因此，皮层成熟的时空模式以分层的方式进行，符合皮层组织的进化根源，感觉运动-联想轴。这个发展计划由来源于多模式人类神经影像学的数据所描述，并与可塑性相关神经生物学事件的层次展开有关。重要的是，这个发育程序用于增强低阶和高阶区域之间的特征变异，从而赋予大脑联合皮层独特的功能特性。然而，越来越多的证据表明，作为人类发育计划拥有属性的晚熟关联皮层的长期可塑性，也赋予了不同发育精神病理学的风险。
		- [[How deep is the brain? The shallow brain hypothesis - 2023]] - 大脑作为皮层深度网络和皮层-皮层下单层连接的混合体
		  collapsed:: true
			- [How deep is the brain? The shallow brain hypothesis | Nature Reviews Neuroscience](https://www.nature.com/articles/s41583-023-00756-z)
			- 深度学习和预测编码架构通常假定神经网络推理是分层的。然而，在深度学习和预测编码架构中，往往忽视了所有分层皮层区域，无论高低，都会直接向皮层下区域投射并接收信号的神经生物学证据。考虑到这些神经解剖事实，目前在深度学习和预测编码网络中以皮质为中心的分层式架构显然是值得怀疑的；这样的架构很可能缺乏大脑所使用的基本计算原则。在这个视角中，我们提出了浅层大脑假说：分层皮层处理与皮层下区域实质贡献的大规模并行过程相结合。这种浅层架构充分利用了皮质微路和丘脑-皮质回路的计算能力，而这些在典型的分层深度学习和预测编码网络中并未包含。我们认为，浅层大脑架构相对于深层分层结构提供了几个关键的好处，并更完整地描绘了哺乳动物大脑如何实现快速和灵活的计算能力。
		- [[Structure–function coupling in macroscale human brain networks - 2024]] - 结构功能关系作为疾病标志
		  collapsed:: true
			- [Structure–function coupling in macroscale human brain networks | Nature Reviews Neuroscience](https://www.nature.com/articles/s41583-024-00846-6)
			- 大脑的解剖结构到底是如何产生一系列复杂功能的仍然不完全清楚。这种从结构到功能的映射的一个有希望的表现是大脑区域的功能活动对底层白质结构的依赖性。在这里，我们回顾了研究结构和功能连接之间宏观耦合的文献，并确定了这种结构-功能耦合（SFC）如何比单独的任何一个特征提供更多关于大脑底层工作的信息。我们首先定义 SFC 并描述用于量化它的计算方法。然后，我们回顾了实证研究，这些研究检查了 SFC 在不同大脑区域、个体之间、在执行认知任务的背景下、随着时间的推移的异质表达，以及它在培养灵活认知方面的作用。最后，我们研究了结构和功能之间的耦合如何在神经和精神疾病中受到影响，并报告了异常的 SFC 如何与疾病持续时间和疾病特异性认知障碍相关。通过阐明大脑结构和功能之间的动态关系在神经和精神疾病存在时如何改变，我们的目标不仅是进一步了解其病因学，而且将 SFC 确立为疾病症状学和疾病的新的敏感标记。认知表现。总体而言，本综述整理了关于神经典型和神经非典型个体的人脑宏观结构和功能之间的区域相互依赖性的当前知识。
-
-
-
-