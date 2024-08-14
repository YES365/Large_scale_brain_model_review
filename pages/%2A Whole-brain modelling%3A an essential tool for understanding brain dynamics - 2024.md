- [Whole-brain modelling: an essential tool for understanding brain dynamics | Nature Reviews Methods Primers](https://www.nature.com/articles/s43586-024-00336-0)
- **全脑建模是一种重要的工具，它为神经科学家提供相关的见解，因为他们正在努力发现健康大脑功能的基本原理。**
- 在过去的几十年里，在理解和诊断神经精神疾病方面进展非常缓慢，主要是由于缺乏对其生物学机制的因果关系的了解。在神经精神疾病中，许多**统计上显著但差异极小**的发现进一步加剧了这种情况。可靠的动物模型很少见，目前治疗方法的问题表明，需要新的研究策略来治疗神经精神疾病。全脑建模是超越这些限制的绝佳工具。计算机模拟全脑模型是使用患者大脑的神经影像学数据设计的，用于识别病理和适当的干预措施。例如，有时患者在昏迷多年后突然意外醒来;个体化计算机全脑模型可用于找到唤醒力，然后可以应用于其他情况。在计算机模拟中，全脑模型必须在**复杂性和真实性之间取得平衡**，以描述体内大脑最重要的功能特征。
- 成功的全脑计算模型从统计物理学中汲取了主导作用，其中哈肯等人的协同学理论解释了远离热力学平衡的开放系统中模式和结构的形成和自组织（[Information and Self-Organization: A Macroscopic Approach to Complex Systems | SpringerLink](https://link.springer.com/book/10.1007/3-540-33023-2)）。这为理解由许多非线性相互作用子系统组成的任何复杂宏观系统的自组织提供了精确的工具。这表明，宏观物理系统遵循独立于其介观成分的定律。每个节点通常由局部神经元动力学的适当近似值组成，可以表示为脉冲神经元网络、平均场模型或介观模型。其基本原理是将解剖结构与功能动力学联系起来（图1）。 解剖结构可以用多种方式表示，理想情况下是通过大规模的束状追踪提供定向解剖连通性的。这些信息需要侵入性束追踪，这在人类中获得是不道德的，因此研究人员使用体内弥散 MRI 结合概率束成像来测量非定向连接。
- **Fig. 1: Principles of whole-brain modelling.**
	- ![Fig. 1](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs43586-024-00336-0/MediaObjects/43586_2024_336_Fig1_HTML.png)
	- 人类神经影像学数据用于全脑模型，将结构解剖学与功能联系起来。全脑模型使用描述该区域行为的人口模型。一些例子是 Ising 模型，该模型适用于大脑活动的二值化表示，通常用于研究相变;Stuart-Landau 振荡器，通常被称为 Hopf 模型，因为它代表了超临界 Hopf 分岔的正常形式，这是一个简单的模型，展示了固定吸引子和振荡之间的参数依赖分岔，通常用于研究区域之间的活动转移;以及动态平均场 （DMF） 模型，这是一种受生物学启发的模型，通常用于寻找上述基于区域的异质测量的区域影响的机械解释。
- 鉴于纤维跟踪很复杂，并且涉及许多步骤，这些纤维重构措施并非没有问题，这些步骤本身就容易受到潜在误差的影响。尽管如此，他们仍然能够将解剖结构与功能动力学联系起来。解剖纤维追踪是在基于结构和功能信息的大脑分区水平上完成的，通常约为 80-1,000 个节点，这进一步降低了建模的复杂性。总体而言，在全脑模型中，功能全局动力学源于局部节点动力学的相互作用，这些相互作用通过潜在的经验解剖学连通性耦合。这种功能活动可以通过全脑神经成像来捕捉，通常是功能性 MRI、脑磁图和脑电图。这些措施捕捉了大脑活动的不同时间尺度。通常，这些模型通过优化参数（包括电导率的异质性）来拟合经验数据。
- 全脑建模框架成功地解释了自发区域间功能活动相关性的模式，形成了功能性MRI捕获的静息态网络。全脑模型可以拟合许多大脑活动的静态和时空测量。这些包括静态功能连通性，但也包括动态测量，如活动波动的时间结构，如功能连通性动态。全脑模型也可用于在更快的毫秒时间尺度上解释大脑活动，但**统一不同的时间尺度仍然存在许多挑战**。至关重要的是，它还被证明，全脑模型可以弥合从毫秒到数十秒的不同时间尺度之间的差距，为全脑活动的多尺度性质提供无与伦比的机制见解。
- 这些个性化的全脑模型可以进一步增强以产生数字孪生。此模型中有不同级别的精度，**应注意不要对精确保真度产生不切实际的期望**。尽管如此，已经有成功的研究，例如，虚拟癫痫患者使用神经影像学数据为癫痫患者大脑的计算机建模提供信息，支持诊断和治疗干预、临床决策和后果预测（[Personalised virtual brain models in epilepsy - The Lancet Neurology](https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(23)00008-X/abstract)）。这表明，癫痫发作的网络级观察是由于神经元或神经群体网络的涌现超同步和高振幅节律状态造成的，这反过来又为个性化治疗提供了潜在的途径。
- 同样，现在正在进行研究，以确定迫使昏迷后患者醒来的最佳刺激目标。使用健康人类参与者的睡眠数据提供了原理证明，并研究了促进从一种大脑状态过渡到另一种大脑状态的准确方法，特别是找到将大脑从深度睡眠唤醒到清醒状态的方法，反之亦然（[Awakening: Predicting external stimulation to force transitions between different brain states | PNAS](https://www.pnas.org/doi/abs/10.1073/pnas.1905534116)）。这是通过使用一个框架来证明的，该框架对构成大脑状态以及可以驱动大脑状态之间转换的内容提供了深入的理解和定量定义。
- 该框架具有很大的临床前景，可以从连接受损的昏迷后患者那里获得结构神经影像学检查。然后，这种连接性可以用于个性化的全脑模型，该模型适用于功能性昏迷后的大脑动力学。然后对全脑模型进行系统探测，以找到对健康大脑动力学的觉醒。随后，成功的觉醒刺激候选者可以使用外部刺激，如深部脑、多焦点经颅直流电或跨磁刺激。鉴于数字孪生在癫痫、多发性硬化症和帕金森病等领域的现有成功，真正觉醒的前景似乎在我们的掌握之中。
- 虚拟大脑是应用最广泛的全脑建模框架。它将各种神经元模型和动力学整合到大脑模拟器中。这种整合将计算建模与多模态神经成像工具无缝地结合在一起，有助于模拟、分析和推断各种脑尺度的神经生理机制。值得注意的是，虚拟大脑可以创建个性化的虚拟大脑，促进对复杂的多尺度神经机制的探索。最近的改进包括通过欧洲平台 EBRAINS 与云服务集成。
- 总体而言，全脑建模因其能够制定解释性机械模型而脱颖而出，从而能够通过非侵入性测量更深入地理解健康的大脑机制。这种能力以前主要与物理学和化学等学科有关，将标志着向更深入的洞察力和对人类认知的未决理解的理解的重大进化飞跃（[The Thermodynamics of Mind: Trends in Cognitive Sciences (cell.com)](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00075-5)）。最近的研究甚至开始表明，热力学和湍流的物理原理如何与全脑模型相结合，可以大大提高我们的理解。因此，计算神经科学已经达到了成熟点，它可以提供工具，最终掌握健康和疾病中大脑功能的基本原理。