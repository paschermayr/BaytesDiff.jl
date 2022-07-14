using BaytesDiff
using Documenter

DocMeta.setdocmeta!(BaytesDiff, :DocTestSetup, :(using BaytesDiff); recursive=true)

makedocs(;
    modules=[BaytesDiff],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/BaytesDiff.jl/blob/{commit}{path}#{line}",
    sitename="BaytesDiff.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/BaytesDiff.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Introduction" => "intro.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/BaytesDiff.jl",
    devbranch="main",
)
