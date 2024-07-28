- [Entropy, free energy, symmetry and dynamics in the brain - IOPscience](https://iopscience.iop.org/article/10.1088/2632-072X/ac4bec)
- #[[Viktor Jirsa]]
-
- ## Abstract
	- 神经科学是概念和理论的发源地，这些概念和理论根植于各种领域，包括信息理论、动态系统理论和认知心理学。
	- 并不是所有这些概念都能被一致地关联起来，有些概念是不可通约的，而领域特定语言对整合构成了障碍。尽管如此，概念整合是一种提供直觉和巩固的理解形式，没有这种理解和巩固，进步仍然是没有指导的。
	- 这篇论文是关于在信息理论框架内确定性和随机过程的整合，将信息熵和自由能与涌现动力学机制和脑网络中的自组织联系起来。我们确定了神经元群的基本性质，从而^^得到一个网络中的等变矩阵^^，其中复杂的行为可以通过流形上的结构化流自然地表示出来，从而建立了与脑功能理论相关的内部模型。
	- 我们提出了一种神经机制，用于从对称性破缺中生成大脑网络连接的内部模型。涌现视角说明了自由能是如何与^^内部模型（内在模式、模态？）^^联系起来的，以及它们是如何从^^神经基质^^中产生的。
-
- ## Introduction
	- ^^预测编码^^是当代最有影响力的脑功能理论之一（[Action understanding and active inference | SpringerLink](https://link.springer.com/article/10.1007/s00422-011-0424-z)）。它基于这样一种直觉，大脑像一个贝叶斯推断系统一样运作，它实现内部生成模型来创造对外部世界的预测。这些预测不断与感官输入进行比较，得出预测误差并更新内部模型(见图1)。从理论的角度来看，大脑功能的预测性编码的表述是引人入胜的，因为它涉及到不同领域的大量深奥的概念。这提供了一个机会，将抽象的概念，如动力学，确定性和随机性作用，涌现和自组织（[The Embodied Mind, Revised Edition | The MIT Press](https://mitpress.mit.edu/books/embodied-mind-revised-edition)），信息，熵，和自由能（[A mathematical theory of communication | Nokia Bell Labs Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/6773024)） ，平稳性，以及更多，关联在一个综合框架内。^^跨领域的集成支持对这种复杂抽象的直观理解，即使通常为了保持易处理性，并不是一个给定概念的复杂性的所有方面都被同等地捕获^^。例如，预测编码理论中的内部模型使用简单的模型(例如在决策中，动力学被简化为转换) ，这很难推广到更复杂的行为。当焦点留在过程的推断部分时，这种方法是完全合理的。在这里，我们希望不这样做，而是实际上强调大脑激活作为内部模型的神经基础，并与复杂行为的涌现的基础理论相关联。尽管如此，考虑到信息理论概念，特别是熵和自由能在预测编码中的重要性，这种努力要求它们与概率分布函数以及相关的确定性和随机性作用相结合，这些都存在于当代大脑网络模型中([Structured Flows on Manifolds as guiding concepts in brain science | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-658-29906-4_6)；[Grand Unified Theories of the Brain Need Better Understanding of Behavior: The Two-Tiered Emergence of Function: Ecological Psychology: Vol 31, No 3](https://www.tandfonline.com/doi/abs/10.1080/10407413.2019.1615207)；[The hidden repertoire of brain dynamics and dysfunction | Network Neuroscience | MIT Press](https://direct.mit.edu/netn/article/3/4/994/95798/The-hidden-repertoire-of-brain-dynamics-and))。
	- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf1_hr.jpg?Expires=1697182079&Signature=mI~81Sqm8~gTz8YHczJv7vQPQTQeE01~wgL1m6UefpjfG8wjpBrGjrYKbe9aXO-yZltVaQ6NNtgK8GZxZOh4iNltA8hEB2cKzOCif6evnzmGrkH47h5-KXOzSVRwcwqNmKteWGloonkj0GhtQtmUCgI3VhQe7hWYyHnDnw77DzVEkk8NRF3-oDNaun0tUVSlWlWTQc1fx8zqS5ZnD0PUfoYezB7vv8GM1iAXYKyLu4OPPohGK7YSYfKcED4xkrNlz8nX2Q316GgKRs63Nu74-8wNdXevUhyIIrZXi2reemWWEjF1R2ydvsClD-85x6SD~En8vbltshk~U5FCluv8NQ__&Key-Pair-Id=KL1D8TIY3N7T8)
		- **Figure 1.**    说明文本中描述贝叶斯大脑假设的基础过程。左边的生成模型，体现在内部的神经动力学中，利用通过感知和行动与外部世界进行的信息交换进行更新，信息交换在贝叶斯推断环路的右侧。
	- ^^自由能^^以前被 Karl Friston 提出作为大脑功能的一个原理（[-* The Free-energy Principle: A Unified Brain Theory? - 2010 - 2685  ](__ The Free-energy Principle_ A Unified Brain Theory_ - 2010 - 2685.md)），它从数学上阐明了自适应的自组织系统如何抵抗自然(热力学)的无序倾向。随着时间的推移，自由能原理已经从亥姆霍兹机器中使用的自由能概念的应用中发展出来，用预测编码来解释大脑皮层的反应，并逐渐发展成为智能体的一般原理，也称为主动推理（[The two kinds of free energy and the Bayesian revolution | PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008420)）。贝叶斯推断过程和最大信息原理都可以有效地重新表述为自由能最小化问题。虽然这些自由能的概念被用在两个相关的框架中，但它们并不完全相同。模糊性可能来自于它们的一般形式类似于热力学亥姆霍兹自由能的事实，但它们遵循两条不同的推理路线(详细讨论见[15])。第一个是所谓的“约束中的自由能”，相当于在最大信息原理的背景下最小化的自由能，代表了确定性约束和随机性作用之间的权衡[7]。正是这种类型的自由能及其约束，是我们在这里的主要考虑。另一个是所谓的“变分自由能”，涉及到贝叶斯大脑假说。这个自由能的概念是通过贝叶斯定律的重新表述而产生的，它将贝叶斯定律重新表述为寻求最小化相对熵的概率分布的优化问题，这个相对熵表示偏离精确贝叶斯后验的误差。
	- 确定性约束在^^流形的结构化流动^^(SFMs)框架中以动力学的形式表示（[Symmetry Breaking in Space-Time Hierarchies Shapes Brain Dynamics and Behavior - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0896627317304142)），这些流形涉及由网络产生的低维动力学系统，因此是大脑理论中内部模型表示的主要候选者。通过概率分布的介导，经验可得的功能（如放电率、能量，方差等等）上的相关性都含有自由能和SFMs两者之间的联系，其中，概率分布在系统中确定性和随机性作用的相互作用下形成。伊利亚 · 普里戈金在他关于熵的意义的论述中阐述了这些力之间的密切关系（[What is entropy? | SpringerLink](https://link.springer.com/article/10.1007/BF00368303)；[The Meaning of Entropy | SpringerLink](https://link.springer.com/chapter/10.1007/978-94-009-3967-7_2)）。在这里，时间的概念超越了重复和退化的概念，而是具有建设性的不可逆转性的概念，具体表现为一个生命系统通过与其所处的环境的熵交换而使自身永久存在。人们认为，生物学需要把时间作为不可逆转的因素刻在物质上。在神经科学的背景下，这让我们想起了英格瓦尔的假设，即大脑通过其时间极化结构模拟“未来记忆”的能力（[Europe PMC](https://europepmc.org/article/med/3905726)），脑维持和导航分布在不同区域的过去、现在和未来的经验。
	- 尽管^^熵^^的数学公式首先出现在经典热力学的背景下，它将热、温度和能量交换等宏观量联系在一起，随后的统计力学将熵公式化为系统处于不同可能微观状态的概率对数的函数。后一种函数形式与香农的熵形式相同，表达式中出现的概率是那些具有不同可能值的变量的概率(参见第2.2节)。在更深层次上，正如埃德温 · T · 杰恩斯(Edwin t Jaynes)在1957年所主张的那样，熵的两个概念被认为是由概率分布表示的不确定性度量的同义词， 在两种情况下，都被视为受可观测量约束的概率分布的预测问题，其中具有最大熵的概率分布是唯一的无偏选择。（[Phys. Rev. 106, 620 (1957)  -  Information Theory and Statistical Mechanics](https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620)；[Phys. Rev. 108, 171 (1957)  -  Information Theory and Statistical Mechanics. II](https://journals.aps.org/pr/abstract/10.1103/PhysRev.108.171)）。
	- 这篇关于熵的意义的简短论文找到了一个由 Hermann Haken [4]建立的协同学的理论框架，它正式地整合了在远离平衡系统中出现耗散结构的数学形式。非线性和不稳定性导致涌现和复杂性的机制，而熵和涨落导致不可逆性和不可预测性。这种动态的性质自然而然地唤起概率概念，这弥补了我们无法精确捕捉系统的个别轨迹。协同学一直是将这些原理转化到其他领域，特别是生命科学和神经科学的驱动力。^^解开确定性和随机性影响的二元性，并阐述它是如何在大脑中产生的，在这里为我们的目标奠定了基础。本次展览的目的是解开后续这些错综复杂的关系，以为大脑动力学建模巩固几个似乎完全不同的框架。^^
	- 考虑到这些概念的互补性，并且为了方便对宏观大脑动力学建模感兴趣的不同背景的受众，我们首先退后一步，回顾一些概率和信息理论的相关基本概念。
-
- ## Information theoretic framework for the brain—modeling system evolution
	- Edwin t Jaynes 强调，信息理论在概念上的巨大进步在于认识到存在一个明确的量，即信息熵，它通过表示一个概率分布的“不确定性的数量”，直观地反映了一个广泛的分布比一个尖锐的峰值表征了更多的不确定性，同时也满足与该直觉一致的其它所有条件（[Phys. Rev. 106, 620 (1957)  -  Information Theory and Statistical Mechanics](https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620)；[Phys. Rev. 108, 171 (1957)  -  Information Theory and Statistical Mechanics. II](https://journals.aps.org/pr/abstract/10.1103/PhysRev.108.171)）。在没有任何信息的情况下，相应的概率分布完全无信息且熵信息消失。在一些确定性的约束条件下，例如物理观测值的平均值的测量，Lagrange的第一类理论允许我们求解在这些约束条件下最大化熵(或等效地最小化自由能 [-* The Free-energy Principle: A Unified Brain Theory? - 2010 - 2685  ](__ The Free-energy Principle_ A Unified Brain Theory_ - 2010 - 2685.md) )的概率分布。从这个意义上说，熵的考虑先于确定性影响的讨论，最大熵分布可以由这样一个事实来断言: 它尊重所有确定性力量的后果，但在其他方面仍然完全不承诺其他影响，如缺失的信息。当熵成为主要概念时，自由能、概率分布函数和 SFM 等相关量之间的关系自然而然地建立起来，通过相关性在现实世界中表达它们自己，通过测量系统状态变量的函数物理上可得，并且原则上允许对系统中的所有参数进行系统的估计。
	-
	- ### Predictive coding and its modeling framework
		- 预测编码讨论的内容及其相关概念，本质上基于三个核心方程（[Free-energy and the brain | SpringerLink](https://link.springer.com/article/10.1007/s11229-007-9237-y)；[Action and behavior: a free-energy formulation | SpringerLink](https://link.springer.com/article/10.1007/s00422-010-0364-z)；[The free-energy principle: a unified brain theory? | Nature Reviews Neuroscience](https://www.nature.com/articles/nrn2787)）：
			- $$p(x,y)=p(y|x)p(x)$$
			- $$\dot{Q}=f(Q,k)+v$$
			- $$Z=h(Q)+w$$
		- 第一个方程建立了贝叶斯定理的简化形式，用概率分布函数 p 表示，其中 p (x，y)是两个状态变量 x 和 y 的联合概率，p (y | x)是给定变量 x 状态的变量 y 的条件概率。^^在贝叶斯框架中，参数和状态变量在某种意义上有着相似的地位，它们都可以用分布来描述，并作为参数输入到概率密度函数中。^^例如，对于给定的参数 k，联合发现状态 x 和 y 的概率写成 p (x，y | k) ，建立获得一组数据 x，y 的可能性，给定一组参数值 k，它们具有 p (k)的先验分布。先验表示我们对模型和初始值的知识。
		- 第二个方程被称为朗之万方程| Lagevin equation（[Action understanding and active inference | SpringerLink](https://link.springer.com/article/10.1007/s00422-011-0424-z)），它建立了神经源水平上的大脑活动由 n 维状态向量 $Q=(x,y,\cdots)\in R^N$ 表示的生成模型方程，$f(Q,k)$ 是依赖状态 Q 和参数 k 或一组多个参数的表示为 m 维流向量 f 的决定性影响。$v\in R^N$ 表示了波动性影响，通常假设为高斯白噪声且具有: $<v_i(t)v_j(t')> = cδ_{ij}δ (t-t’)$，其中 δij 为 Kronecker-delta，δ (t-t’)为 Dirac 函数。关于噪声影响的更一般的公式，包括乘性或有色噪声是可能的，我们在这里建议读者参考相关文献。
		- 第三个方程建立观测器模型，通过正向模型 h (q)和测量噪声 w 将源活动 Q(t) 与实验可获得的传感器信号 Z(t)联系起来。对于脑电图测量，h 是由 Maxwell 方程建立的增益矩阵|  gain matrix  ; 对于功能性磁共振成像测量，h 是由神经血管耦合和血流动力学 Ballon-Windkessel 模型给出的。在本文中，观测器模型不受关注，尽管它在现实世界的应用中具有极大的重要性，并且在模型反演和参数估计问题中经常扮演一个主要的污染因素的角色。我们在这里提到这些工程问题是出于完备性的考虑，但是为了简单起见，现在假设 h 是具有零测量噪声的恒等运算，因此 Z = Q。
		- 预测编码涉及大量行为神经科学中的研究领域（[Dynamic Patterns | The MIT Press](https://mitpress.mit.edu/books/dynamic-patterns#:~:text=His%20core%20thesis%20is%20that%20the%20creation%20and,multistability%2C%20abrupt%20phase%20transitions%2C%20crises%2C%20and%20intermittency.%20)） ，特别是致力于感知-行动和动力系统的科学研究的生态心理学| ecological psychology （[Perceiving, Acting and Knowing | Toward an Ecological Psychology | Rob](https://www.taylorfrancis.com/books/mono/10.4324/9781315467931/perceiving-acting-knowing-robert-shaw-john-bransford)；[Ecological laws of perceiving and acting: In reply to Fodor and Pylyshyn (1981) - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0010027781900020?via%3Dihub)）。在这里，詹姆斯 · 吉布森强调了环境的重要性，特别是有机体的环境如何向其提供各种行为的感知。这个知觉-行动的循环与预测编码中内部生成模型的预测和更新的循环非常匹配。生态心理学强调的一个特殊的细微差别是，生态可利用的信息，而不是外围或内部的感觉，导致了感知-行动动力学的出现。Scott Kelso 和他的同事对这个框架的形式化做出了重大贡献，并开发了实验范例来测试有机体和环境之间协调的内部模型的动力学性质（[Outline of a general theory of behavior and brain coordination - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608012002286)）。这些范式在理论上受到 Hermann Haken 的协同学的启发，而协同学是自组织理论的基础。它产生了大量的研究工作，主要集中在知觉和行为状态之间的转换，包括双手和多感觉运动协调的建模（[A theoretical model of phase transitions in human hand movements | SpringerLink](https://link.springer.com/article/10.1007/BF00336922)；[Modeling Rhythmic Interlimb Coordination: Beyond the Haken–Kelso–Bunz Model - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0278262601913107)）和系统实验测试（[APA PsycNet](https://psycnet.apa.org/record/2010-22326-001)）。这些方法随后被推广到更大范围的范例（[APA PsycNet](https://psycnet.apa.org/record/2014-31650-002)；[Time Scale Hierarchies in the Functional Organization of Complex Behaviors | PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002198)），其目标是提取生物体与环境相互作用的行为的主要特征。这些大量的工作为行为的动态描述的好处提供了实质性的证据，这些可以归纳到 SFMs 的框架中，作为行为（[APA PsycNet](https://psycnet.apa.org/record/2014-31650-002)）和大脑（[A quantitative model of conserved macroscopic dynamics predicts future motor commands | eLife](https://elifesciences.org/articles/46814)）中的功能性知觉-行动变量。
	-
	- ### Maximum information principle
		- 我们对于确定状态变量 x 的离散值 xi的确定性过程和随机过程的理解通过计算相应的概率分布pi来获得。香农证明了一个重要的事实，即存在一个信息量 h (p1，... ，pn) ，它唯一地度量由这些概率分布表示的不确定性的数量。在他最初的证明中，他表明了三个基本条件的要求，尤其是连接事件和概率的合成定律，自然导致了下面的表达：
			- $$H(p_1,\cdots,p_n)=-K\sum_ip_i\ln p_i$$
		- 其中 k 是正常数。由于度量 H 直接对应于统计力学中熵的表达式，因此它被称为信息熵。一般来说，对其性质的讨论是从观察者可获得的信息量相关的概率规范的角度出发的。拉普拉斯的不充分推理原则认为，在没有任何区分信息的情况下，两个事件发生的概率相等。^^这种主观的思想学派把概率看作是人类无知的表现，形式化地表达了我们对事件发生与否的期待^^。这种思维对于预测编码理论及其对大脑产生的认知过程的解释是基本的。^^客观的思想学派源于物理学，认为概率是事件的客观属性，原则上，概率可以通过随机实验中事件的频率比来衡量^^。在这里，通过研究确定性和随机性过程基础上的预测编码，我们希望采取一个中间的立场。确定性和随机性作用的分析使用客观学派的语言，而预测编码框架内的后续解释是主观学派的一部分。预测编码中的生成模型通过朗之万方程表征了两种类型的力量，并塑造了概率分布的形状，然后通过经验测量函数 <g(x)> 提供给我们获得这些信息的途径。三角形的括号表示期望值：$$<g(x)>=\sum_ip_ig(x_i)$$
		- 这些 g (x)的相关性，连同归一化要求$\sum_ip_i=1$，表达了确定性和随机性影响所提供的约束，在这些约束下，熵有一个最大值。 任何其他的赋值，除了熵的最大值之外，都会引入其他确定性的偏差或者任意的假设，而这些假设是我们所没有的。这种洞察力是 这种洞察力是德 Edwin T Jaynes 的最大信息原理的精髓。因此，我们在这些约束条件下通过引入Lagrange参数 λi 使熵 h 最大化，得到：
			- $$p_i=exp(-\lambda_0-\sum_j\lambda_jg(x_i))$$
		- 其中，概率分布的表达式被推广为事件 xi 的多个测量函数 gj (xi)。该分布的熵 s 最大，为：
			- $$S_{max}=\lambda_0+\sum_j\lambda_j<g_j(x)>$$
		- 经验测量和最大熵之间的联系是由配分函数 Z 建立的：
			- $$Z(\lambda_1,\cdots,\lambda_n)=\sum_iexp(-\sum_j\lambda_jg_j(x_i))$$
			- $$<g_j(x)>=-\frac{\partial}{\partial\lambda_j}\ln Z,\lambda_0=\ln Z$$
		- 经验测量所获得的相关性通过这些方程与固定的概率分布函数相联系，从而与通过朗之万方程表达的生成模型(即决定性力)相联系。在下面的部分，我们将提供明确的神经科学相关的例子。
-
- ## Emergence in self-organizing system—modeling system dynamics
	- 上述知觉-行动动力学和预测编码框架互补地解释了大脑和行为是如何通过对环境和自我的作用和观察而被塑造的，而 SFMs 框架则提供了这些塑造过程的最终结果是如何体现的机械图景。SFMs 为持续演化的内部生成模型的抽象相空间的时空基础结构提供了概念化| 在抽象相空间为持续演化的内部生成模型提供了时空基础结构的概念化，以相互协调的神经活动的自发涌现的流形的形式，大脑在这些流形上体验和改造世界。我们现在简要地讨论在大脑动力学的背景下研究SFMs的潜在的核心动力系统的概念和相关的数学机制。
	-
	- ### Time-scales separation
		- 自组织系统的涌现要求一些典型的低维吸引子的稳定性发生改变。所考虑的系统是高维 N 自由度非线性系统。在这些自由度张成的空间中，每个点都是状态向量，代表整个系统的潜在状态。随着时间的发展，系统的状态会发生变化，从而在状态空间中描绘出一条轨迹。系统遵循的规则可以理解为引起状态向量变化的力，这些力可以定义流。为了使这个系统能够产生低维行为，即 M 维，$M\ll N$，必须有一个机制能够将高维空间中的轨迹指向低维的 M 维子空间。在数学上，这转化为两个与不同时间尺度相关的流分量: 首先，低维吸引子空间包含一个流形 M，它吸引快速时间尺度上的所有轨迹; 其次，在流形上有一个结构流 F(.)规定了一个缓慢时间尺度上的动力学，这里的缓慢是相对于快速动力学向着吸引流形的崩塌过程的，如图2所示。为了紧凑和清晰，假设系统的状态是由 n 维状态向量 Q(t)在时间 t 的任意给定时刻描述的。然后我们将全部的状态变量分解为 q 和 s 两部分，其中 q 中的状态变量定义了与低维子空间(功能网络)中涌现行为相关联的 M 个任务特定变量，而 s 中的 N-M 个变量定义了剩余的自由度。当然，N 要比 M 大得多，而且变量q的子空间中的流形必须满足一定的约束条件才能局部稳定，在这种情况下，所有的动力学都被吸引到这里。
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf2_hr.jpg?Expires=1697182079&Signature=LRmmYsOTB-Z3qCN17Vg-sv4DtlsXXdbqN9CGWVvYOWsPsSJpyCMCx1~dMnv8APB1-sAqhpQ6ljtybb1G-9G10cwv-KZprlFlyT6eoLXJv8wn6NAA1KoOrjyCWRphHQ3JDFgBJFcb6f6t8i2l~~wl9smSYT4kMCSYI2jw8GTAb-mITsyCKlH5AJ~hPWlr35mhhK6z~kkMPgme8sywuADitS2IDLQCOY0za3uqgNkO42W4DTBz~EfZXCCK8ZMLdJ7pZaH9wjpqbrjZ2FyWdjweURBfMOqzTXdm0jdp2ARYndDg4-DwGbJo64S7v6cVMmjydi7n4QDA7sAZHtL52h5H6g__&Key-Pair-Id=KL1D8TIY3N7T8){:height 619, :width 873}
			- **Figure 2.**     自组织动力系统的涌现。外部输入推动系统脱离由控制参数k控制的平衡。系统中的非线性相互作用导致在控制参数的临界值处出现少量宏观模式(序参数)。剩余的自由度被奴役，并遵循序参量的演化。
		-
	- ### Synergetics
		- 我们首先简要概述了协同学的一些概念，这将有助于我们更好地理解 SFM 的概念。协同学是远离热平衡的开放系统(即那些通过物质、能量和/或信息流与环境接触的系统)中自组织模式形成的理论，这些系统由无数弱相关的微观元素构成。由于微观元素之间的相互作用，这些系统可能在空间上和时间上形成有序的模式，这些模式通常是宏观的，可以用数量有限的所谓序参量(或集体变量)来描述。当一个宏观模式失去稳定性而另一个稳定模式占据主导时，大脑活动模式的自发转换(即非平衡相变)就会发生。系统状态(或相位)的稳定性意味着，如果因扰动离开了某个状态，系统将趋向于回到那个状态。当系统失去稳定性时，系统反而倾向于从那个状态转移到另一个稳定状态。接近这些(宏观)不稳定点时，返回的时间大大增加。因此，宏观状态对于扰动的反应相当缓慢，而底层的微观组分保持其各自的时间尺度。因此，它们动力学的时间尺度大不相同(时间尺度的分离)。从缓慢演化的宏观状态的角度来看，微观成分变化如此之快，以至于它们能够瞬间适应宏观变化。因此，即使宏观模式是由子系统生成的，前者也可以，比喻地，奴役后者[4]。有序状态总是可以用很少的变量来描述(至少在分叉点附近) ，因此，原来的高维系统的状态可以用几个甚至一个集合变量——序参量来概括。序参量便张成工作空间。序参量和被奴役的微观分量之间的循环关系，这种关系产生序参量，有时被称为循环因果关系，这有效地允许对系统的动力学性质进行低维描述（[Reconstruction of the spatio-temporal dynamics of a human magnetoencephalogram - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/016727899500226X)）。循环因果关系的概念和自组织系统中低维动力学的涌现是赫尔曼 · 哈肯协同学的核心[4]。
		- 协同学的数学形式可以简述如下。一个 N 维状态向量被定义为时间的函数 $Q(t)\in R^N$ ，它包含系统的所有状态变量。状态向量的演变被一个非线性常微分方程描述：
			- $$\dot{Q}=F(Q,\{k\})+v(t)$$
		- 其中 F 是一个非线性函数，可以捕捉所有可能的相互作用，{k}是一组控制系统状态的控制参数，并且是时间无关的。非线性演化方程包括噪声的影响 υ (t) ，这将在后面考虑，但不是在这里。状态向量及其演化方程也可以推广到包含状态变量的空间相关性，但这里只考虑空间离散系统就足够了。详细的数学处理见哈肯(1983)。
		- 协同学的自组织和涌现的切入点是宏观层面上系统状态的质变(状态转换)。一个给定的状态通常对应于状态空间中的一个(静止的)固定点 Q0或周期的循环(极限环)(见图2) ，协同学的目的是描述其在状态空间中的邻域的流(图3)。
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf3_hr.jpg?Expires=1697182079&Signature=Hn2-AhE50JSt4yVcVOMjSpZWQ8Oou9KqZ8poYKsBvz9s7SNRMxk4gUA3yj3FTt39C7YbqyBHv6zh-0wA66JAyXfi0jQ8CJmqANOFdbhkigfnyQTFPLrhLhlx-mDvcQHIyLpU0kwWiZaRBX9TvUUR-zivlGjcv4UkoYc32kNCEtOPE-U6bPvL5C7ybHFWK~tlnHkr27KMXWxZVCToDgjHobMtsraDqao9IiWMtgO2RVOA2y7hAGzoK0-aj40tomSEhQGbbmhM~jadkQA10xdwulr8RjlKBXxhRRtGBexvnRZ0lEyy3AfpXwVfYxOZ~E6VTo3a5ODzZ5LAnjeXUSYbbQ__&Key-Pair-Id=KL1D8TIY3N7T8)
			- **Figure 3.**     协同学与 SFMs。在自组织，涌现模式被限制在低维流形，如球面。当流局部(左边，矩形区域)改变其拓扑结构时，协同学中的临界性出现。临界性在 SFMs 是非局部的，当整个流形的流为零时就会出现。
		- 与稳态 Q0的偏差ε(t)为 Q(t) = Q0 + ε(t) ，它的时间演化可以用 F 在稳态 Q0附近的 Taylor 展开来近似：
			- $\dot{Q}=F(Q,\{k\})=F(Q_0,\{k\})+L(Q_0,\{k\})\varepsilon+A(Q_0,\{k\})\varepsilon:\varepsilon+\cdots$
		- L、A分别是泰勒展开式的一阶和二阶项。当控制参数发生变化时，系统在参数空间中受到引导，会遇到导致分岔的参数配置，从而导致当前状态的稳定性发生变化，系统在宏观层面上发生重大重组。分岔点定义了工作点 k0，因此定义了第二个局部约束，这次是在参数空间中，紧邻状态空间中的工作点 Q0。在这些局部条件下，雅可比矩阵L的特征值 λ0至少有一个变为零，从而允许局部中心流形定理的应用。它指出一个新变量的小子集$q(t)\in R^M,M\ll N$，即序参量，这些序参量主宰系统动力学，并通过时间尺度分离的方法奴役剩余变量，其中序参量 q(t)是慢的，奴役变量 s(t)是快的。用协同学的术语来说，奴役是指变量 s(t)的内在动力学可以绝对消去，其长期演化表示为序参量的函数，即 s(t) = s (q(t))。这就导致了对涌现动力学的简化描述，如下所示：
			- $\dot{q}=\lambda_0q+P(q,s(q),\{k_0\},Q_0),q(t)\in R^M,M\ll N$
			- $s(t)=s(q(t)),s(t)\in R^{N-M}$
		- 其中 p 是被k0,Q0参数化的非线性函数，通过分解为序参数和被约束变量，可以从 f 解析地计算出 p。
	- ### Structured flows on manifolds arising from symmetry breaking
		- 虽然协同学在概念上并不局限于状态空间中的局部工作点，但实际上情况一直如此。在生物学中，这对协同框架的效用造成了巨大的限制，因为探索状态空间中的流似乎是生物体的一项基本活动。为此，对称性提供了另一个指导原则来定义参数空间中的工作点（[Symmetry Breaking in Space-Time Hierarchies Shapes Brain Dynamics and Behavior - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0896627317304142)；[Structured Flows on Manifolds as guiding concepts in brain science | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-658-29906-4_6)），目的是克服局限于状态空间局部区域的局限性。具有对称性的动力系统称为等变动力系统（[The Symmetry Perspective | SpringerLink](https://link.springer.com/book/10.1007/978-3-0348-8167-8)） ，其中群论方法提供了一种自然语言。为了证明这一点，让我们考虑方程(1)的状态变量正式表示为 q(t) 而不是 Q(t) ，原因是。我们在第4.2节的末尾回到这一点。进一步，为了简单起见，我们只考虑一个控制参数 k，设 γ 是作用于其解的群。如果 f 与 γ 的群作用相交，则方程为 γ 等变方程，即对于所有 $\gamma\in\Gamma,F(\gamma\cdot q)=\hat{\gamma}$。对于临界控制参数值 k0，应该存在这种对称性。Γ 等变的一个重要结果是，如果解 q(t)是常微分方程的解，那么对于所有 γ ∈ γ，γ · q (t)也是。如果对称性是连续的，即 γ · q = q + δq，则对应的群是李群，其元素在状态空间中具有 m 维光滑流形的拓扑，其群运算是元素的光滑函数。然后，方程(1)的稳态解跨越一个光滑流形 m，该流形 m 定义为
			- $\dot{q}=F(\gamma\cdot q_0)=F(q_0+\delta q)=\hat{\gamma}\cdot F(q_0)=0$
		- 允许一个连续的位移 δq 沿着流形。在对称性破缺的情况下，k = k0 + μ，其中 μ 很小，系统的解可以通过完全对称解的扰动来近似.
			- $$\begin{equation}
			  \dot{q} = F(q, \{k\}) = \underbrace{F(q_0, k_0)}_{=0} + \underbrace{\left. \frac{\partial F}{\partial q} (q, k) \right|_{q_0, k_0}}_{M} \cdot (q - q_0) + \underbrace{\left. \frac{\partial F}{\partial k} (q, k) \right|_{q_0, k_0}}_{N} \cdot \underbrace{(k - k_0)}_{\mu} + \cdots
			  \end{equation}
			  $$
		- 其中 M 是平稳解的光滑不变流形。一个零特征值与 M 的切空间相关联，该切空间建立了与其他自由度相关的时间尺度层次结构，这是协同学通常所知道的。对于完全对称，k = k0，流形上的所有点是稳定不动点，如果 m 是稳定的。对于小对称性破缺，μ something 1，一个缓慢的流动出现沿歧管，这是相对于正交于歧管的快速动力学缓慢。这两种情况如图4所示，用于圆形流形 m: 0 = 1-x2-y2。
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf4_hr.jpg?Expires=1697182079&Signature=oJhFbxes5K4BuKJCVfz6A5rrEjg1gyfQLCNpef6laSytSQJVMOnXhoM8FPD0j0cp5SFQWiJ0HXtHjLxJNzFvFP5Jr3FBVjR46b9QV2hLAqgOKugS~36LqPwHF1NFXddfVWIhPX56P7AhLo4R2ywHD1Un2w2jeoIaVU9QJvqUkf34rGfAQRe6VoQdzV5S2TGdM~slSejXVV20uPRkBMnpMjEo3MwJYndEO1yvB4ju-Z9t3uDMKYLyVhRvo6Lb4yWD3CvgvKTcEKtdd~4VLudFnIPaCjmHeGcBMetdIXSMex6~fQSBlxFWn35JECFoglVIE8SS8SeFk5Trw-jaHoWKRA__&Key-Pair-Id=KL1D8TIY3N7T8)
			- **Figure 4.** 圆形流形上的结构化流图示。上：由于两个子系统完全对称，x,y方向上零流的两个零线重叠。圆形流形由稳定不动点组成。下：对称性破缺，如通过x和y的耦合形成，使得两个零线移开，在流形上形成一个由稳定和不稳定固定点组成的结构化流的狭窄通道。
		- 流形上的流结构完全由对称性破缺通过 n 中的 k 决定。在神经科学的背景下，例如在大规模的脑网络模型中，这样的对称性破缺是由于大脑区域之间的连接或者个别区域的局部特性的变化所致。从协同学的传统框架中我们知道，没有与之相关的巨大维度减化，导致序参数从 n 维压缩到 m 维。相反，对称性必须在一个子空间范围内定义，然后由其余的 n-m 变量完成，就像传统的协同框架一样。假设这些变量在临界点 k = k0处没有通过不稳定性，那么它们可以用通常的方法通过绝热消去，它们的动力学表示为序参数 q (t)的函数。传统的协同框架与 SFMs 的区别在于，前者认为全 n 维系统允许自然的全维度减化，而后者则需要假设或施加额外的约束。这些考虑使我们得出以下形式的系统方程:
			- $$
			  \begin{gather}
			  \dot{q} = M(q, s, \{k_0\}, Q_0) \cdot (q - q_0) + \mu N(q, s, \{k_0\}, Q_0) \\
			  s(t) = s(q(t))
			  \end{gather}
			  $$
		- 其中 $$q(t) \in \mathbb{R}^M, M \ll N, \text{ and } s(t) \in \mathbb{R}^{N-M}$$ 这组方程建立了由SFMs表示的基本数学框架，接下来我们将在该框架内发展出基本网络方程。
-
- ## Equivariant dynamics in the brain
	- 涌现SFM模型的先例讨论是关于满足约束条件的自然界所有动力学系统的一般性讨论。这一部分使动态系统与神经科学相关的链接，目的是说明 SFMs 如何自然地从基本的神经科学网络中产生，并在状态空间中创造概率分布。
	- ### Neural mass models show basic 2D dynamics
		- 神经质量模型是神经元集体活动的简化数学表示。它们通常来源于一群神经元，这些神经元由耦合点神经元模型表示。在假设动作电位分布和/或神经元间耦合的统计特性的前提下，应用平均场理论推导出集体变量方程， 捕捉种群的均值、方差和更高阶的统计矩的演化过程。值得注意的例子包括 Brunel Wang 模型假设 Poisson 分布尖峰[63] ，Zerlaut 等模型使用主方程和传递函数公式[64] ，Stefanescu-Jirsa 模型[65,66]利用神经元参数的异质性导致同步神经元簇（[A Low Dimensional Description of Globally Coupled Heterogeneous Neural Networks of Excitatory and Inhibitory Neurons | PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000219)；[Neural Population Modes Capture Biologically Realistic Large Scale Network Dynamics | SpringerLink](https://link.springer.com/article/10.1007/s11538-010-9573-9)；[Phys. Rev. X 5, 021028 (2015)  -  Macroscopic Description for Networks of Spiking Neurons](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.021028)；[Just a moment...](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP272317)）。从理论的角度来看，Montbrio 等人的平均场推导特别有吸引力，因为它在洛伦兹假设下是精确的。它导出了两个集体状态变量，平均放电频率 r 和平均神经元膜电位 v。相应的相流在图5(a)中绘制，方程在图5(b)中绘制。这些和所有其他神经质量模型的共同点是，他们降低了平均场动态到一个低维表示，往往在二维。神经质量动力学通常包括一个下降状态对应的低点火率，一个上升状态对应的高点火率，以及能力显示振荡动力学普遍在北部。暂时忽略振荡动力学，我们可以从概念上减少共存的上升和下降状态及其通过分支的稳定性的变化，到一个单一变量，x 的相位，如图5(c)所示。这里的相流被简化为两个稳定状态的分离，在外部控制参数 k 的变化下，这两个稳定状态可能通过鞍点分岔而失去稳定性。该模型在数学上与Wong–Wang模型一致，后者是由Brunel Wang模型在隔绝近似的条件下推导得到。我们想要强调的是，这种表示并不是Brunel Wang神经质量模型特有的，而是刻画了所有神经质量模型的基本动力学特性，即对于系列中间控制参数，同时存在低、高放电状态，经控制变量取高/低值的分岔，丢失低/高放电状态。因此，在这我们用该简化神经质量作为下节等变脑网络模型的基本构建块。
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf5_hr.jpg?Expires=1697182079&Signature=Og4OJDyLX-wIAljccWqh0m50AbYRMlq22eE98mpbHd1q-o8uflaSKn8M688R21krFomvIcVvdWQLtwLLKmIpTL7XmVc7mYRiyXEAGTUZBqv2TmZko1K1ib~oABq-wvhMECPSR7xWwdTPxAaSCWiqrFDIxzjMX8h3c3xkueOfJatQwVjYcsGjacvZjbC7qWXiTeQpO-krAJyjIfeLWfmbsk9q~haXy4apdgI65J5nB5RknVo76pVA3KRkT7VNxuHZ9jGh6KBFMglCs4QL6ksezkdTyVKHaQYp5M0tdtEhEMuB~Muver8YP~XXvdoW~o6Nu6qjBdk8Bg3PPpXu3q7ArQ__&Key-Pair-Id=KL1D8TIY3N7T8)
			- **Figure 5.**  化神经群体模型到基本的一维形式。（A）不同输入强度的二维Montbrio模型。（B）Montbrio神经质量模型的组成。（C）简化的一维神经质量模型。
	- **Derivation of the equivariant matrix**
		- 在一个由双稳态神经元群组成的网络中，对称性破缺自然导致了 SFMs 的产生。这可以理解为以下几点。让我们首先考虑一个直观的玩具网络的两个节点的状态变量 x 和 y。基本方程是这样的：
			- $$
			  \begin{align*}
			  \dot{x} &= f(x,k) + v = x(1-x^2) + k + v \\
			  \dot{y} &= f(y,k) + v = y(1-y^2) + k + v
			  \end{align*}
			  $$
		- 形成一个耦合神经质量模型系统，其中 k 是局部兴奋性，v 是噪声，遵循方程(1)的记号。图6显示了这种情况下状态空间中的相流，说明了四个稳定不动点、一个不稳定不动点和四个鞍点。红线和绿线表示零斜线。上述方程现在可以正式改写为：
			- $$
			  \begin{aligned}
			  \dot{x} &= f_x(x,y,G,k) + \mathbf{v} = x(1-x^2-y^2) + Gy^2x + k + \mathbf{v} \\
			  \dot{y} &= f_y(x,y,G,k) + \mathbf{v} = y(1-x^2-y^2) + Gx^2y + k + \mathbf{v},
			  \end{aligned}
			  $$
		- 其中 rhs 上的第一项表示前面3.3节中看到的相同的循环不变流形。对于 g = 1，这些方程与解耦系统相同。当 g 从1到0变化时，零线的形状从无穷大(直线)到椭圆形不断变化，变成闭合的圆形(见图6)。G 作为第二个控制参数，量化互连度的程度，从完全不连通(g = 1)到全对全耦合(g = 0)拓扑。对于一个高度互联的网络，g ≈0，这些中间值约束了一个封闭的流形上的相流，以原点为中心，创造了一个对称的流动和时间尺度的分离。如果引入其他形式的对称性破缺，比如 k 的区域变化或者不对称的连接，那么这就提供了一种系统地控制流的方法。
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf6_hr.jpg?Expires=1697182079&Signature=eVEGpz5OvajHEgB68Wm6kJORnWZUNhNOq-OY8pCXXfghUQukzESilHZj4ZQMXcVmiBo978f75Gn9lkaCSGhZJqazAWBEpi7aBV3WdsX3a29x63GwGKQlTfq~-0Qatylzcsk9G99Nxy1NIL8TKP4aTIXOVCUa5qYAQVjKPzZyRGq-rgSIAv-PgeS3LjBQ0r6LNh-EYP9W8GrarBsecVzgSM9QyM7WOeZkLpT9PRc1TcqBT~9NNXp7jb6Ocl~McOdGDMAOgmqCetCUxBI2lFUv-IXkT-YKvLuGHKzcsI0D78J-yrbAzVhrK3NZDmHd-HTQ6lYUaj-9duqI2-vc9oi~mA__&Key-Pair-Id=KL1D8TIY3N7T8)
			- **Figure 6.**    耦合参数G变化时，两个节点的状态空间流。从左上角起：G=0,0.25,0.75,1。
		- 将这个网络扩展到三个网络节点，方程如下：
			- $$
			  \begin{aligned}
			  \dot{x} &= f(x,y,z,G,k) + v = x(1-x^2-y^2-z^2) + G(y^2+z^2)x + k + v \\
			  \dot{y} &= f(x,y,z,G,k) + v = y(1-x^2-y^2-z^2) + G(x^2+z^2)y + k + v \\
			  \dot{z} &= f(x,y,z,G,k) + v = y(1-x^2-y^2-z^2) + G(x^2+y^2)z + k + v
			  \end{aligned}
			  $$
		- 其中对应的相流在图7所示的状态空间中绘制为 G = 1。这种情况与前面示例中的未连接网络节点相对应。同样，原点是一个不稳定的不动点，对称地位于一个由1个不稳定的不动点，8个稳定的不动点组成的立方体中，由12个鞍点分隔，沿着定义立方体边缘的零点对齐(见图7)。由于三个节点之间的连通性正在建立，G 向 G = 0的完全连通网络缩小到较小的值。对于后者，不变流形是一个2球体，以原点为中心，半径为1，流量为零。
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf7_hr.jpg?Expires=1697182079&Signature=Q9QM8KwOry4eUX6dET6EYf-HGO0qEHzZbnyVrxVWrEBzdhM6YPQ1bEiQbyRBpRw-HpS6ksVTTfN0ciIYZxbmw~0oxZytERiwPZgr~xBMVTR7X-XNRk5dFGB-Pis1kghSkn~iF59wElnsHdjR1MoQZqJ4qIN-Pok1yOa-rn9arsWS62qalAzX74-KQS3NHm~AqyZrFxhYmagaw4juEJ1IRrRxG4ezpNtAmsE9HZ7Jrgq1kzC2C1fRe14J3v4xiuWuLvUfXRBj-ueO-SlFNm3Ca4HiY9fmr3JtfnG5uapPJRKG2XNMrINBMbgWQV0~0qKn68SXTrfYd-MPOHHbQroq4Q__&Key-Pair-Id=KL1D8TIY3N7T8)
			- **Figure 7.**    由三个不耦合节点组成的网络中的流。一个三维立方体跨越一个等变矩阵，该矩阵中心有1个不稳定不动点(绿色) ，8个稳定不动点(红色)和12个鞍点(绿色)。上图和下图以不同的细节程度显示相同的情况。
		- 这种情况可以形式化地推广到由 n 个基本单元组成的网络 ：
			- $$
			  \dot{x}_i = f(x_i, x_j, G, k_i) + v_i = x_i \left(1 - \sum_{j=1}^N x_j^2\right) + G \sum_{j\neq i} c_{ij}x_j^2x_i + k_i + v_i.
			  $$
		- 该网络在 n 维状态空间中建立了一个等变矩阵，在不存在任何耦合的情况下，存在2n 个稳定不动点，由相同数目的不稳定不动点分开，这些不动点都以原点为中心。随着连接的建立，系统的相流越来越局限于(n-1)-球的接近，这是流形 m，如第2.2节所讨论的，完全由稳定不动点组成，对于 g = 0，cij = 1，完全类似于二维和三维情况。在更复杂的对称性破缺情况下，例如通过引入一个连接体，在这里 cij ≠1，或者区域兴奋性的变化和局部噪声 vi，在这个流形上可以建立大范围的结构化流。
		- 基于连接体的网络模型已经广泛应用于静息状态活动（[Multistability in Large Scale Models of Brain Activity | PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004644)；[Modular slowing of resting-state dynamic functional connectivity as a marker of cognitive dysfunction induced by sleep deprivation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1053811920306418)；[Dynamic Functional Connectivity between order and randomness and its evolution across the human adult lifespan - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S105381192030642X)），但是除了描述性统计(如功能连接性、功能连接性动力学、多尺度熵)以及假设的机制(次临界性、随机共振)之外，还没有提供严格的理论视角。同变矩阵通过连接体的对称性破缺增加了一个有吸引力的替代解释。然而，它仍然缺乏一个重要的解释性论据，因为没有实质性的由等变矩阵对称性破缺提供的维度减化。不变流形保持 n-1维。正如在3.3节中所讨论的，现在这里专门针对神经科学，它提出了一些问题，如何实现从N维变量Q到M维变量q的维度简化，因为 m 应该比 n 小得多。
		- 以下是一个可能的实现。在更现实的神经科学模型中，固定点不会完全相同。一般来说，下行状态比上行状态更稳定。虽然这些差异将不可避免地将梯度到引入网络的流中，但它不足以减少低维流形，这是任务特定的过程所需要的，并且已知存在于人类大脑活动中。文献中的讨论唤起了去相关作为信息处理的重要机制的可能性（[Temporal decorrelation by SK channels enables efficient neural coding and perception of natural stimuli | Nature Communications](https://www.nature.com/articles/ncomms11353)） ，并且在这里可以通过上状态的振荡来实现。通过频率分离或平均解耦是动力系统理论(例如旋转波近似)中众所周知的机制，并在大规模的大脑网络中用于组织同步（[Transmission time delays organize the brain network synchronization | Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2018.0132)）和交叉频率耦合（[Frontiers | Cross-frequency coupling in real and virtual brain networks | Frontiers in Computational Neuroscience](https://www.frontiersin.org/articles/10.3389/fncom.2013.00078/full)）。这种振荡行为的能力是一个优秀的候选者，目的是将一个等变的 M 维流形与它的 N-M 维互补子空间分离开来，提供了一个通往未来的科学研究的明确的道路。
		-
	- ### Derivation of the probability distribution function and the free energy
		- 自由能反映了由系统的确定性特征产生的流形上结构流的演化过程。在大规模脑网络模型的背景下，我们证明了通过连接体的对称性破缺可以作为这些特征涌现的候选机制，从而建立了与生物物理过程(如Hebbian学习和其他可塑性机制)的联系。 然后探索流形的随机波动以熵的形式推动系统发展  。流形上的每个点都与一个概率分布相关联。作为没有随机力的极限情况，这些分布是近似 δ 函数的尖峰，结构流是完全确定（[Symmetry Breaking in Space-Time Hierarchies Shapes Brain Dynamics and Behavior - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0896627317304142)）。这些确定性和随机性影响之间的联系是福克-普朗克方程，它规定了概率分布函数的时间演化。通过将我们的注意力限制在对其平稳特性的预测上，统计特性的解释是与时间无关的，我们可以参考状态，否则系统在时间 t 时的状态的解释只能在时间 t 时进行的测量的基础上进行。
		- 我们希望通过两个例子来说明这一点。第一个示例适用于前面在4.1节中讨论的情况，其中两个节点是耦合的。让我们把初始焦点放在两个稳定的固定点中的一个。我们将其流线性化，得到如下表达式  ：
			- $$
			  \begin{align*}
			  \dot{x} &= f_1(x,y) = -x + 2\beta y + v = -\frac{\partial V}{\partial x} + v \\[10pt]
			  \dot{y} &= f_2(x,y) = -(y - y_0) + 2\beta x + v = -\frac{\partial V}{\partial y} + v
			  \end{align*}
			  $$
		- 其中V(x,y)表示势，v表示噪声，y0表示偏移量，β表示耦合强度。确定和随机影响在实验获得的相关性中表现出来，建立在构造概率分布函数p(x,y)时需要满足的约束。函数g(x,y)的期望值是<g(x,y)>，可以由统计动量<x>,<y>,<x^2>,<y^2>,<xy>,...等表示。平稳概率分布函数
			- $$p(x,y)=Ne^{-2V(x,y)/Q}$$
		- 是Fokker–Planck方程的时间独立解（[Nonlinear, Nonequilibrium Landscape Approach to Neural Network Dynamics | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-61616-8_15)）。
			- $$\dot{p}+\sum_{i}\frac{\partial}{\partial x_i}(f(x,y)-\frac{1}{2}Q\sum_i\frac{\partial p}{\partial x_i})=0$$
		- 其中，x1=1,x2=y,N是标准化常数。根据假设可表示为：
			- $$p(x,y)=exp(\lambda_0+\lambda_1x^2+\lambda_2x+\lambda_3xy+\lambda_4y+\lambda_5y^2)$$
		- 其中所有的拉格朗日乘数λi可以通过实验获得的相关性<g(x,y)>和归一化条件$$\int_{-\infty}^{\infty}p(x,y)dxdy=1$$显式地估计出来。不失一般性，假设λ1=-1，λ2=0，λ3=-2β，λ4=2y0，λ5=-1，上式可简写为：
			- $$p(x,y)=Cexp(-x^2+2\beta xy+2y_0y-y^2)$$
		- 归一化因子是从具有λ0的项中获得的。这个联合概率分布函数形式上可以重写以反映贝叶斯定理同时联系到我们最初的讨论，即
			- $$p(x,y) = p(x|y)p(y) = N_x \exp(-(x - \beta y)^2/Q) N_y \exp(-((1-\beta^2)y^2 + 2y_0y - y_0^2)/Q),$$
		- 其中Nx,Ny是$$p(x|y),p(y)$$的归一化常数。如果两个节点是独立的，那么自然地动量分解<xy>=<x><y>和确定性耦合β=0。条件概率p(x|y)变成独立于y，即p(x|y)=p(x)，并且p(x)和p(y)都是完全高斯的。表达式$$F=(x-\beta y)^2/Q$$表示自由能，直观地将节点之间的相互作用刻画为静态概率密度的变形，如图 8 所示
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf8_hr.jpg?Expires=1697182079&Signature=aHWJNnQ34AjEjPaAyRMM4fIF6Ycqceu~p8xtp0ogwOzmbQm3cbYurytZA~casDxn5mnAxF84uTGP~X30XPQPaLMZZeOKDDvRKo08gh2Wdg6VU1gw4Q4WUoSkjrPtdSrCiJeYAwnlAldEnoEeNtc6MYY6xQzWPPcwjNqJTMtQDfjWTmwMgPlTv5T8c~kRxtDBIS4PpCmU1x3zDLoaaGoUwvOJU36kGv-qzh1XD7DYvcnyqnzJyQM-s5WhJYu8COU3kpsuO94w3MCFfvTrNNH0O4TMBioP0v7WZ7mBr4guvhU0xHsyy81uUR~kjftiDlA3cmwAPOoy-85SE2vI2xvNCg__&Key-Pair-Id=KL1D8TIY3N7T8)
			- **Figure 8.**     耦合强度为β=1-G的二维线性耦合网络中，同一节点稳定不动点周围的概率分布。上：β=0,下：β=0.5。
		- 第二个例子包括4.2节中的等变矩阵。为了简单起见，我们将讨论再次限制在两个节点上。按照与前面示例相同的数学步骤，耦合节点的势能函数读取
			- $$V(x,y) = -\frac{1}{2}y^2 + \frac{1}{4}y^4 - 2ky - \frac{1}{2}x^2 + \frac{1}{4}x^4 - 2kx + \frac{1}{2}\beta x^2y^2$$
		- 其中β=1-G表示耦合强度，β=0时表示没有耦合。同前面一样，可把平稳概率函数写为p(x,y)=p(x|y)p(y)，其中 ：
			- $$
			  \begin{aligned}
			  p(y) &= N_y \exp\left(-\left(\frac{1}{2}y^4 + y^2 + 4ky\right)/Q\right) \\[10pt]
			  p(x|y) &= N_x \exp\left(-\left(-\frac{1}{2}x^4 + x^2 + 4kx - \beta x^2y^2\right)/Q\right).
			  \end{aligned}
			  $$
		- 完全类比，自由能由$$F=-\left(-\frac{1}{2}x^4 + x^2 + 4kx - \beta x^2y^2\right)/Q$$明确给出，同时给出Friston自由能原理的说明示例，其中x和y之间相关性的确定性和随机性这一基本特性通过交互作用项β来刻画。β=0，这两个节点在统计上是独立的，并且每个节点都显示出图 9 所示的双峰分布。每个单独的固定点可以用高斯局部近似，这可以通过固定点附近全概率密度的Taylor表达式解析推导出来。随着耦合强度β向1增加，平稳概率分布改变其形状，接近圆形流形并在流形上构造其结构化流。不同β值的分布如图9所示。这里的结构化流包括四个稳定和四个不稳定的不动点。
		- ![](https://content.cld.iop.org/journals/2632-072X/3/1/015007/revision2/jpcomplexac4becf9_hr.jpg?Expires=1697182079&Signature=dmo3hyllmWiqlk-ktS4AGMxx1ckb2t7QRQ5tLXGEzvCm4glnR5lgT1XUxzg~8Jk7fc884paWDfYEBOzZC4wlGWBHANxlXKRC0hP9G~LUNtr98Gh62J-dtGKoLs50Dgd7Xfk6Dx43PNWkbHd1TZugtNqanTsthM6O7tTlzr6P2wv2eo8KRfCANt2VyqruzcanEykz6CCEiT9ujh3hI7i26zI8uG9BehfbSHjstVCcSoJSy6uEXC9mPT0rDtsYh7YDoLi3mh4umAz9s1o1yKK~w4lqQl~4DvzDB1zuqDxXRwTjJpEVW3x5BPSNN4MxzQlE4R~aSLQ16UUStTElRzcybQ__&Key-Pair-Id=KL1D8TIY3N7T8)
			- **Figure 9.**    双节点等变网络中的势和平稳概率分布。从左到右耦合强度为β=1-G=0,0.75,1。上面一行为势，底下一行为平稳概率分布。
- ##
- ## Final thoughts and conclusions
	- 如果我们接受这里讨论的熵和信息概念作为第一个基本原语，那么从信息理论框架自然而然地推导出的，结合我们对熵作为不确定性的直观理解，就是概率分布函数形状变化中表现出的确定性和随机性影响的基本组织。这一结果适用于自然界中的所有系统，而不仅仅是大脑，这就是为什么Hermann Haken经常将其称为协同学的第二个基础[4]。
	- 将范围缩小到大脑网络中存在的力量，我们将神经团和网络的基本特性与状态空间中不变流形的出现联系起来，这些流形是行为神经科学中已知的结构化流的载体，特别是生态心理学和协调动力学。结构化流形（SFMs）代表了预测编码中的内部模型。这种与行为的联系很重要，因为它经常被用来指导神经科学研究，使其具有生态意义。当明确计算概率分布函数时，自由能自然地表现为流形上的结构化流，而这种流又是由网络节点之间的耦合产生的。在主动推断的过程中，大脑调整这些耦合以改变相应的SFMs（即内部模型）。
	- 我们要记住，这些耦合（或更准确地说：耦合参数）一方面是预测编码中自由能最小化下的变异目标，另一方面负责实现行为中特定任务的功能架构。与解决网络在连接性和参数方面"如何"演变的机制的自由能原理不同，SFMs则解决"什么"——"即，需要满足什么样的网络约束才能使特定的流和流形出现"[16]。因此，熵和自由能可以用来解释通过学习和发展过程的演变，但也可以看作在认知表现的较短时间尺度上发挥作用。相应地，低维任务特定流形上的流在抽象状态空间中捕捉了熵作为大脑中建设性不可逆性的机制表现，因此，作为神经活动和行为之间的主要使能链接。
-