# Fall-Detection using LSTM Autoencoder v1.2.1
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/ParthenopeDeepTeam/Fall-Detection-using-LSTM-Autoencoder">
    <img src="media/background.png" alt="Logo" width="400" height="240">
  </a>

  <h3 align="center">Abstract</h3>

  <p align="justify">
Even if it's not well known, falling is one of the causes for accidental injury for most of the elderly people over 65 years old. If the fall incidents are not detected in a timely manner, they could lead to serious injury. Moreover, such incidents can happen in outdoor environments, such as parking lots, where the probability of being noticed and receive punctual help is even smaller. A system which detects abnormal events (such as falls) only using camera streams, without requiring extra sensors, could provide timely aid by triggering an alarm. In this work, we provide a solution using an LSTM Autoencoder which is one of the most used machine learning algorithm for anomaly detection. After applying a real-time pose estimation framework, called OpenPose, to the real-time video source, the poses are preprocessed by different steps in order to be normalized and filtered, and then processed by an LSTM Autoencoder. The model is trained to learn the normal behaviour of a walking person through a large dataset of 19 joint points of human body. If the autoencoder generates an output which is too different from the corrispondent input, it means that the time window that was given to the model is an anomaly, meaning the person is falling.
    <br />
</p>

<h3 align="center">Reconstruction error</h3>
<p align="center">
    <img src="https://github.com/ParthenopeDeepTeam/Fall-Detection-using-LSTM-Autoencoder/blob/main/media/reconstructions.gif" width="480">
    <br>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#proposed-models">Proposed Models</a></li>
      </ul>
      <ul>
        <li><a href="#frameworks">Frameworks</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

In this work we propose a system that monitors the pose of people in a scene only through a traditional camera, and learns the way the user behaves. The system would learn patterns over time and would therefore be capable of building a solid knowledge of what a normal pose behaviour is, and what it isn't (in particular, falling events).
Both train set and test set were made ad-hoc for this work, in order to build a dataset that represents a specific task: video surveillance in places like pedestrian areas, parking lots and parks of various type. So video footages are setted up to capture a large area, with the camera installed at about 3 meters of heigth.

### Proposed models

Researches show that LSTM Autoencoders and similar models led to promising results in detecting anomalous temporal events in a semi-supervised manner. The idea of training a model with only "normal events" is important because, in nature, abnormal instances occur very rarely, therefore the acquirement of such data is expensive.
Hence the base kind of model used for this study is an Autoencoder with LSTM layers as its elementary units as showed in the figure below. Throughout the model selection phase various model architectures have been tried and tested in order to find the best one.
<p align="center">
    <img src="https://github.com/ParthenopeDeepTeam/Fall-Detection-using-LSTM-Autoencoder/blob/main/media/architecture.png" width="530">
    <br>
</p>

### Frameworks

* [Keras/Tensorflow 2.0](https://www.tensorflow.org/)
* [Python 3.7](https://www.python.org/downloads/release/python-370/)
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```


<!-- CONTACT -->
## Contact

* Andrea: [![LinkedIn Andrea][linkedin-shield]][linkedin-andrea-url]
* Antonio: [![LinkedIn Antonio][linkedin-shield]][linkedin-antonio-url]


* Project Link: [https://github.com/ParthenopeDeepTeam/Fall-Detection-using-LSTM-Autoencoder](https://github.com/ParthenopeDeepTeam/Fall-Detection-using-LSTM-Autoencoder)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/ParthenopeDeepTeam/Fall-Detection-using-LSTM-Autoencoder/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/ParthenopeDeepTeam/Fall-Detection-using-LSTM-Autoencoder/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-andrea-url]: https://www.linkedin.com/in/andrea-lombardi/
[linkedin-antonio-url]: https://www.linkedin.com/in/antonio-junior-spoleto/
