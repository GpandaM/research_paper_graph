# Literature Review

## Introduction

 The field of multiphase flow, which involves the simulation and analysis of the behavior of multiple immiscible fluids or fluid-solid mixtures, has gained significant attention in recent years due to its broad applications in various industries such as oil and gas production, chemical processing, and environmental engineering. In this context, this comprehensive analysis focuses on research trends and developments in multiphase flow dynamics over the past decade (2012-2025). The temporal scope of research shows a gradual increase in publications from 2012 to 2025, with a significant surge in interest around 2020. Key trends during this period include the application of advanced numerical methods such as Eulerian-Lagrangian approaches and machine learning techniques, as well as the investigation of multiphase phenomena at the nanoscale. Notable developments include studies on the dynamics of particle laden flows, the role of interfacial properties in multiphase systems, and the application of topological data analysis to characterize complex fluid behavior. As we delve deeper into this analysis, we will explore these trends and developments in greater detail, providing insights into current research advancements and potential future directions in the domain of multiphase flow dynamics.

## Research Areas and Themes

 The given research output focuses on the numerical simulation of electrical deformation and coalescence in water-in-oil emulsions using phase-field (PF) and arbitrary Lagrangian–Eulerian (ALE) methods. This study lies within the broader research areas of Computational Fluid Dynamics (CFD), Electrohydrodynamics, Two-Phase Flow Simulation, Water-in-Oil Emulsions, Crude Oil Demulsification, and Electrostatic Separation Technologies.

1. Main research themes and clusters:
The primary themes of this study revolve around the development and validation of numerical models for simulating electrical deformation and coalescence in water-in-oil emulsions. The research is clustered into understanding time-dependent droplet deformation, coalescence thresholds, and electrohydrodynamic interactions.

2. Interdisciplinary connections:
The interdisciplinary nature of this study lies at the intersection of fluid mechanics, electrical engineering, and computational modeling. The research connects various aspects of these fields to investigate the electrohydrodynamics of water-in-oil emulsions and their deformation and coalescence behaviors.

3. Emerging areas of focus:
The emerging areas of focus within this field include investigating the applicability of these numerical approaches in broader contexts, such as other types of emulsions or multiphase flows. Moreover, advancements in high-performance computing and data analysis techniques may lead to more accurate and efficient simulations.

4. Geographic or institutional concentrations:
Geographically, research in this area is concentrated in countries with significant oil and gas industries, such as the United States, Russia, and Saudi Arabia. Institutionally, research is conducted at universities and research institutions specializing in fluid mechanics, computational modeling, and energy-related research. Collaborative efforts between these institutions and industry partners are also common to advance the state of the art in this field.

## Methodological Approaches

 The provided data represents a diverse range of research studies conducted in the domain of computational fluid dynamics (CFD), focusing primarily on various methods used to model complex fluid flows. Upon analyzing this data, several key observations can be made regarding the methodological approaches employed and their evolution over time.

Firstly, a dominant trend in the CFD research community is the use of hybrid Lagrangian-Eulerian methods for simulating multiphase flows, where Eulerian methods are used to model well-defined fluid regions, while Lagrangian methods are employed to track the behavior of discrete entities such as droplets or interfaces. This approach has gained popularity due to its flexibility in modeling complex physical phenomena with minimal computational overhead.

A second observation is the evolution of methods over time, which shows a progression towards more sophisticated approaches. Early studies utilized simple Eulerian models for simulating fluid dynamics, while later works introduced multiphase and hybrid methods to model more intricate flow behaviors. Furthermore, recent developments have focused on improving computational efficiency and accuracy by employing advanced meshless numerical techniques and artificial intelligence-based models.

In terms of methodological trends, there is a growing interest in developing methods that can accurately capture free surfaces and fluid-structure interactions (FSI), as evidenced by the increasing number of studies using Lagrangian-Eulerian approaches or direct Arbitrary Lagrangian-Eulerian (ALE) schemes. Additionally, there has been a focus on improving the numerical stability and accuracy of various methods, such as implementing adaptive mesh methods and advanced reconstruction techniques like Weighted Essentially Non-Oscillatory (WENO).

Despite these advances, there are still limitations to some of the more complex methods, including increased computational demands and potential for numerical instability. The choice of an appropriate method depends on the specific application and available resources. It is essential for researchers to understand the strengths and limitations of various approaches to effectively model fluid dynamics in various contexts.

## Key Contributions and Findings

 Title: Advances in Computational Fluid Dynamics: Breakthrough Discoveries, Theoretical Frameworks, Methodological Innovations, Practical Applications, and Influential Works

1. Breakthrough Discoveries and Innovations:
In recent years, significant advancements have been made in the field of computational fluid dynamics (CFD), leading to new discoveries and innovations. One groundbreaking discovery is the application of Eulerian-Lagrangian hybrid solvers in external aerodynamics for modeling airfoil stall [1]. These solvers accurately predict static and dynamic stall, providing insights into complex flow phenomena that have important implications for aircraft design. Another remarkable achievement is the development of a Lagrangian split-step method for viscoelastic flows [2], which effectively captures viscoelastic behavior and large deformations while offering advantages over mesh-based solvers in handling complex fluid interfaces.

2. Theoretical Contributions and Frameworks:
Theoretical contributions have played a crucial role in advancing the field of CFD. For instance, the development of the Arbitrary Lagrangian-Eulerian (ALE) method for compressible multi-material flows on adaptive quadrilateral meshes [3] provides a direct solver with sharper shock capturing capabilities and improved solution accuracy in high-gradient regions. Additionally, advances in electrohydrodynamic simulations using both phase-field and ALE methods [4] have demonstrated physical correctness and quantitative results for modeling these complex processes.

3. Methodological Advances:
Methodological advances have significantly impacted the field of CFD. For example, the hybrid Lagrangian-Eulerian Particle Finite Element Method for free-surface and fluid-structure interaction problems [5] maintains accuracy while reducing computational cost through adaptive remeshing and dynamic adjustment of Lagrangian regions. Furthermore, the one-way coupling approach in landslide-generated wave simulations using coupled multi-phase flow and Boussinesq-type models [6] significantly reduces computational costs without sacrificing wave propagation accuracy.

4. Practical Applications and Implementations:
The practical applications of CFD have grown exponentially in various industries, including aerospace, automotive, energy, and biomedicine.

## Future Research Directions

 Future research directions in the domain of computational modeling of electrical deformation and coalescence of droplets in water-in-oil emulsions can be explored as follows:

1. Unresolved Research Questions and Gaps: Although phase-field (PF) and arbitrary Lagrangian–Eulerian (ALE) methods have been compared for simulating electrical deformation and coalescence of droplets, further investigation is required to understand the optimal conditions for their application in various industrial settings. The impact of additional factors such as electric field strength, droplet size distribution, and flow rate on droplet behavior needs to be studied.
2. Emerging Opportunities and Challenges: With advancements in computational power and parallel computing techniques, it is possible to extend the applicability of these numerical methods to larger-scale industrial processes involving complex geometries and multiple interacting phases. However, significant challenges remain in accurately capturing the intricacies of electrohydrodynamic interactions and topology changes during coalescence.
3. Technological Opportunities: The development of hybrid models combining aspects of both PF and ALE methods could provide a more versatile platform for simulating complex phenomena involving droplet deformation, coalescence, and electrohydrodynamic interactions in industrial applications such as crude oil demulsification and electrostatic separation technologies.
4. Interdisciplinary Research Potential: The study of electrical deformation and coalescence of droplets has implications for various disciplines, including materials science, chemical engineering, and fluid mechanics. Interdisciplinary collaborations can lead to a more comprehensive understanding of these phenomena and the development of novel technologies.
5. Practical Applications and Societal Impact: The computational modeling of electrical deformation and coalescence of droplets has significant practical applications in various industries, such as energy, food processing, and pharmaceuticals. The advancement of accurate numerical methods can lead to the optimization of industrial processes and potentially contribute to the development of sustainable technologies with reduced environmental impact.

## References

[1] ['Linxu Fan, Floyd M. Chitalu, Taku Komura'] (2025). A Hybrid Lagrangian–Eulerian Formulation of Thin-Shell Fracture.
[2] ['Pasolari, R.', 'Ferreira, C.S.', 'van Zuijlen, A.', 'Baptista, C.F.'] (2024). Dynamic Mesh Simulations in OpenFOAM: A Hybrid Eulerian–Lagrangian Approach.
[3] ['Zhihao Qian, Tengmao Yang, Moubin Liu'] (2024). An Overview of Coupled Lagrangian–Eulerian Methods for Ocean Engineering.
[4] ['Jingyuan Zhang, Corinna Schulze-Netzer, Tian Li, Terese Løvås'] (2024). A Novel Model for Solid Fuel Combustion with Particle Migration.
[5] ['Thomas Lesaffre, Antoine Pestre, Eleonore Riber, Bénédicte Cuenot'] (2024). Correction Methods for Exchange Source Terms in Unstructured Euler-Lagrange Solvers with Point-Source Approximation.
[6] ['Cheng Fu, Massimiliano Cremonesi, Umberto Perego'] (2024). A Hybrid Lagrangian–Eulerian Particle Finite Element Method for Free-Surface and Fluid–Structure Interaction Problems.
[7] ['Arnida L. Latifah, Novan Tofany, Mochammad Raja Jaefant Alphalevy'] (2024). Landslide-Generated Wave Simulation Using Coupled Multi-Phase Flow and Boussinesq-Type Models.
[8] ['Guo, L. Cheng, H. Yao, Z. Rong, C. Wang, Z. Wang, X.'] (2024). CFD–DEM method is used to study
the multi‑phase coupling slag
discharge fow feld of gas‑lift
reverse circulation in drilling shaft
sinking.
[9] ['Xiaolong Zhao, Shicang Song, Xijun Yu, Shijun Zou, Fang Qing'] (2024). An Arbitrary Lagrangian-Eulerian Discontinuous Galerkin Scheme for Compressible Multi-Material Flows on Adaptive Quadrilateral Meshes.
[10] ['Vladimir Chirkov, Grigorii Utiugov, Petr Kostin, Andrey Samusenko'] (2024). Physical Correctness of Numerical Modeling Electrohydrodynamic Processes in Two-Phase Immiscible Liquids Basing on the Phase-Field and Arbitrary Lagrangian–Eulerian Methods.
[11] ['R. Pasolari, C. J. Ferreira, A. van Zuijlen'] (2024). Eulerian–Lagrangian hybrid solvers in external aerodynamics: Modeling and analysis of airfoil stall.
[12] ['Ahmed Mostafa Elsayed Mohamed Borg'] (2024). CFD Investigations on Multiphase Flow in Well-Control Operations.
[13] ['Martina Bašić, Branko Blagojević, Branko Klarin, Chong Peng, Josip Bašić'] (2024). Lagrangian Split-Step Method for Viscoelastic Flows.
[14] ['Sylwia Polesek-Karczewska, Paulina Hercel, Behrouz Adibimanesh, Izabela Wardach-Święcicka'] (2024). Towards Sustainable Biomass Conversion Technologies: A Review of Mathematical Modeling Approaches.
[15] ['W. Düsterhöft-Wriggers, S. Schubert, T. Rung'] (2024). A Two-Phase Volume of Fluid Approach to Model Rigid-Perfectly Plastic Granular Materials.
[16] ['Stefan Christian Endres, M.'] (2023). A Discrete Differential Geometric Approach for Simulation of Coupled Multiphase Mesoporous Systems.
[17] ['Victor Chéron, Jorge César Brändle de Motta, Thibault Ménard, Alexandre Poux, Alain Berlemont'] (2023). A Coupled Eulerian Interface Capturing and Lagrangian Particle Method for Multiscale Simulation.
[18] ['Yu-Hsuan Huang, Yang-Yao Niu'] (2023). Development of a Lagrangian–Eulerian Approach-Based Five-Equation Two-Fluid Model for Simulation of Multiphase Reactive Flows.
[19] ['Hongwei Jia, Fengyong Lv, Liting Xu, Yanming Kang, Yunfeng Wang, Xin Xiao'] (2023). CFD Modeling of Two-Phase Flow with Surfactant by an Arbitrary Lagrangian–Eulerian Method.
[20] ['L. X. Zhou'] (2023). An Eulerian–Eulerian–Lagrangian Modeling of Two-Phase Combustion.
[21] ['Yu Lv, John Ekaterinaris'] (2023). Recent Progress on High-Order Discontinuous Schemes for Simulations of Multiphase and Multicomponent Flows.
[22] ['Peter Böhling, Johannes G. Khinast, Dalibor Jajcevic, Conrad Davies, Alan Carmody, Pankaj Doshi, Mary T. Am Ende, Avik Sarkar'] (2023). Computational Fluid Dynamics-Discrete Element Method Modeling of an Industrial-Scale Wurster Coater.
[23] ['N. Matos, M. Gomes, V. Infante'] (2023). Numerical Modelling of Soft Body Impacts: A Review.
[24] ['Boyang Chen, Bruño Fraga, Hassan Hemida'] (2023). A three-phase Eulerian–Lagrangian model to simulate mixing and oxygen transfer in activated sludge treatment.
[25] ['Jingyuan Zhang, Tian Li, Henrik Ström, Boyao Wang, Terese Løvås'] (2023). A Novel Coupling Method for Unresolved CFD-DEM Modeling.
[26] ['Dzmitry Misiulia, Praveen Kumar Nedumaran, Sergiy Antonyuk'] (2023). Effect of the Discharging Flap on Particle Separation in a Cyclone.
[27] ['Mohamadali Mirzaei, Sønnik Clausen, Hao Wu, Sam Zakrzewski, Mohammadhadi Nakhaei, Haosheng Zhou, Kasper Martin Jønck, Peter Arendt Jensen, Weigang Lin'] (2023). CFD Simulation and Experimental Validation of Multiphase Flow in Industrial Cyclone Preheaters.
[28] ['Behrad Esgandari, Stefanie Rauchenzauner, Christoph Goniva, Paul Kieckhefen, Simon Schneiderbauer'] (2023). A comprehensive comparison of Two-Fluid Model, Discrete Element Method and experiments for the simulation of single- and multiple-spout fluidized beds.
[29] ['Tuo Wang, Fengshou Zhang, Jason Furtney, Branko Damjanac'] (2022). A review of methods, applications, and limitations for incorporating fluid flow in the discrete element method.
[30] ['Stefan Christian Endres, Marc Avila, Lutz Mädler'] (2022). A Discrete Differential Geometric Formulation of Multiphase Surface Interfaces for Scalable Multiphysics Equilibrium Simulations.
[31] ['Georg Hammerl'] (2022). A Multipurpose Euler-Lagrange Framework for the Numerical Simulation of Particle and Dispersed Flow Problems.
[32] ['Patrick Kopper, Stephen M. Copplestone, Marcel Pfeiffer, Christian Koch, Stefanos Fasoulas, Andrea Beck'] (2022). Hybrid Parallelization of Euler–Lagrange Simulations Based on MPI-3 Shared Memory.
[33] ['Peng Zhao, Ji Xu, Qi Chang, Wei Ge, Junwu Wang'] (2022). Euler-Lagrange simulation of dense gas-solid flow with local grid refinement.
[34] ['Mohamadali Mirzaei, Peter Arendt Jensen, Mohammadhadi Nakhaei, Hao Wu, Sam Zakrzewski, Haosheng Zhou, Weigang Lin'] (2022). A hybrid multiphase model accounting for particle agglomeration for coarse-grid simulation of dense solid flow inside large-scale cyclones.
[35] ['Amal Sahai, Grant Palmer'] (2022). Variable-Fidelity Euler–Lagrange Framework for Simulating Particle-Laden High-Speed Flows.
[36] ['Miguel Masó, Alessandro Franci, Ignasi de-Pouplana, Alejandro Cornejo, Eugenio Oñate'] (2022). A Lagrangian–Eulerian Procedure for the Coupled Solution of the Navier–Stokes and Shallow Water Equations for Landslide-Generated Waves.
[37] ['Nikos Spyropoulos, George Papadakis, John M. Prospathopoulos, Vasilis A. Riziotis'] (2022). Assessment of a Hybrid Eulerian–Lagrangian CFD Solver for Wind Turbine Applications and Comparison with the New MEXICO Experiment.
[38] ['Mauro Murer, Giovanni Formica, Franco Milicchio, Simone Morganti, Ferdinando Auricchio'] (2022). A Coupled Multiphase Lagrangian-Eulerian Fluid-Dynamics Framework for Numerical Simulation of Laser Metal Deposition Process.
[39] ['Cagatay Guventurk, Mehmet Sahin'] (2022). A Mass Conserving Arbitrary Lagrangian–Eulerian Formulation for Three-Dimensional Multiphase Fluid Flows.

