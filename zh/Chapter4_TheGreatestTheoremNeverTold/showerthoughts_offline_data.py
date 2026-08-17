"""
离线版的 "Reddit r/Showerthoughts 抓取脚本" 替代品。

原书用 top_showerthoughts_submissions.py 通过 praw(Reddit 的官方 Python API
封装库)实时抓取 r/showerthoughts 板块的帖子及其点赞/点踩数据。这需要联网、
需要 Reddit API 凭据(client_id/secret,写在 praw.ini 里),在没有网络访问权限
或没有配置好 API 凭据的环境下(比如本仓库构建时所在的沙箱环境),原脚本会
直接抛出异常,导致整个 notebook 无法端到端运行。

本脚本是它的一个离线替代品,目的是让 notebook 不需要联网就能完整跑通:

  - `top_post`: 一个字符串, 模拟"锚点帖子"的标题(对应原脚本 `%run ... N`
    调用里, 跳过前 N 条之后遇到的那条帖子)。
  - `contents`: 长度为 N 的字符串列表, 每个元素是一条模拟的帖子标题。
  - `votes`: 形状 (N, 2) 的 NumPy 数组, 每一行是 [upvotes, downvotes]。

**重要说明**: 下面的帖子文本全部是本脚本用固定随机种子人工生成的、风格模仿
"Showerthoughts"(那种"洗澡时突然想到的冷知识/脑洞")版块的*占位文本*,
并不是真实抓取自 Reddit 的内容,也不归属于任何真实用户 —— 这是为了在离线
环境下如实地复现原书这个例子的统计结构(点赞/点踩比例、样本量分布等),
而不是伪造或冒充真实的 Reddit 数据。如果你有网络访问权限和自己的 Reddit
API 凭据,可以直接换回原书的 top_showerthoughts_submissions.py。

赞成票/反对票的生成方式模仿了 Reddit 真实的偏态分布: 绝大多数帖子只有很少
的投票数(因此比例样本噪声很大),少数几条"爆款"帖子会积累成百上千票
(因此比例样本更可信)——这正是本章"如何给 Reddit 帖子排序"这个例子想要
说明的核心统计现象,合成数据也刻意保留了这个结构,不影响后续的贝叶斯排序
分析在教学上的意义。
"""
import sys

import numpy as np

_TEMPLATES = [
    "The {adj} {noun} you {verb} as a kid probably still {verb2} {adv}.",
    "We never see a {noun} get {verb} in real time, only the {adj} result.",
    "Every {noun} you've ever {verb} is now somewhere in a landfill.",
    "The {adj} version of you from ten years ago would be {adv} confused by today.",
    "Your phone knows more about your {noun} than your {adj} friends do.",
    "A {noun} is just a {adj} {noun2} that hasn't {verb2} yet.",
    "Somewhere a {noun} is having the {adj} day of its {noun2}.",
    "You've forgotten more {noun} than most people will ever {verb}.",
    "The {adj} smell of {noun} always brings back {adv} specific memories.",
    "Nobody teaches you how to {verb} a {noun}, you just figure it out {adv}.",
    "The first person to {verb} a {noun} had no idea it would become {adj}.",
    "Most {noun} are just {adj} {noun2} in a trench coat.",
    "If {noun} could talk, they'd probably be {adv} tired of us.",
    "You are the {adj} {noun} of somebody else's {noun2}.",
    "Every {adj} {noun} was once somebody's {adv} risky idea.",
]
_ADJ = ["weird", "tiny", "loud", "quiet", "ancient", "modern", "invisible", "sticky",
        "blue", "heavy", "fragile", "sacred", "forgotten", "accidental", "eternal"]
_ADV = ["secretly", "quietly", "eventually", "briefly", "constantly", "rarely",
        "suddenly", "endlessly", "barely", "mysteriously"]
_NOUN = ["sandwich", "umbrella", "keyboard", "cloud", "shoelace", "elevator",
         "houseplant", "calendar", "mirror", "backpack", "doorbell", "sock",
         "traffic light", "toothbrush", "streetlamp"]
_NOUN2 = ["idea", "machine", "ritual", "habit", "mistake", "invention", "memory",
          "coincidence", "experiment", "tradition"]
_VERB = ["touched", "broke", "lost", "built", "borrowed", "painted", "renamed",
         "forgot", "fixed", "carried"]
_VERB2 = ["works", "exists", "matters", "helps", "waits", "lingers", "counts"]


def _make_title(rng):
    t = rng.choice(_TEMPLATES)
    return t.format(
        adj=rng.choice(_ADJ), adv=rng.choice(_ADV), noun=rng.choice(_NOUN),
        noun2=rng.choice(_NOUN2), verb=rng.choice(_VERB), verb2=rng.choice(_VERB2),
    )


def load(n_submissions=100, seed=42):
    """返回 (top_post, contents, votes),含义见文件头的说明。"""
    rng = np.random.default_rng(seed)

    contents = [_make_title(rng) for _ in range(n_submissions)]

    # 模仿真实 Reddit 的偏态投票分布: 总票数服从对数正态分布
    # (少数帖子票数极多, 大多数票数很少), 真实的赞成票比例
    # 服从一个整体偏向"正面"但仍有相当离散度的 Beta 分布。
    total_votes = np.clip(rng.lognormal(mean=3.0, sigma=1.6, size=n_submissions).astype(int), 1, None)
    true_ratio = rng.beta(6, 2, size=n_submissions)

    upvotes = np.zeros(n_submissions, dtype=int)
    downvotes = np.zeros(n_submissions, dtype=int)
    for i in range(n_submissions):
        ups = rng.binomial(total_votes[i], true_ratio[i])
        upvotes[i] = ups
        downvotes[i] = total_votes[i] - ups

    votes = np.array([upvotes, downvotes]).T
    top_post = contents[0]
    return top_post, contents, votes


if __name__ == "__main__":
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    top_post, contents, votes = load()
    print(f"(离线合成数据, 与 top_showerthoughts_submissions.py <N> 的行为等价, N={n_sub} 未使用, 仅为保持调用方式一致)")
