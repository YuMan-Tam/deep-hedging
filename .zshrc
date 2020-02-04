# Set bin paths
PATH_brew="$HOME/.local/brew"
export PATH="$HOME/.local/bin:$PATH_brew/bin:$PATH_brew/sbin:$PATH"
export PATH="$PATH_brew/opt/coreutils/libexec/gnubin:$PATH"

# The existing LD_LIBRARY_PATH defined by administrator interfered 
# with brew.
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$HOME/.local/intel/compilers_and_libraries_2019/linux/mkl/lib/intel64"

# Set alternative location for temporary files because /tmp is often 
# full.
export TMPDIR=$HOME/.cache

# Export default editor.
export EDITOR="$PATH_brew/bin/vim"

# Shortcut to compress file into tar.xz.
function tarxz(){
    if [ -n "$1" ] && [ -z "$2" ]; then
        tar cfJ "$1.tar.xz" "$1"
    elif [ -n "$1" ] && [ -n "$2" ]; then
        tar cfJ "$1.tar.xz" "$1.$2"
    fi
} 

#######################################################################
# Tensorflow
#######################################################################

# Important to experiment with numactl options!

# bsub -I numactl -C 0 -m 0 python [FILE]

export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export TF_DISABLE_MKL=0
export MKL_DISABLE_FAST_MM=1


export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
export KMP_BLOCKTIME=0

#######################################################################
# Linuxbrew Configuration
#######################################################################
 
# Run brew install -v -interactive to debug.
# Run "git fetch origin" and "git reset --hard origin/master"
# to reset formula.
export HOMEBREW_CACHE="$HOME/tmp/brew-pkgs"
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_GITHUB_API=1
export HOMEBREW_TEMP="$HOME/tmp/tmp"

export MANPATH="$MANPATH:$PATH_brew/manpages"

#######################################################################
# Java Configuration
#######################################################################

# Switch between different versions of JAVA via "brew unlink" and 
# "brew link --force". Modify also JAVA_HOME.
export PATH="$PATH_brew/opt/openjdk@11/bin:$PATH"
export JAVA_HOME="$PATH_brew/opt/openjdk@11"

#######################################################################
# Zsh Configuration
#######################################################################
# Debug tools when there is performance issue: set -x

ZSH_AUTOSUGGEST_USE_ASYNC=1
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE="fg=8"

#######################################################################
# Oh-My-Zsh Configuration
#######################################################################
export ZSH=$HOME/.local/brew/Cellar/oh-my-zsh/20190701

ZSH_THEME="dallas"

DISABLE_AUTO_UPDATE="true"
COMPLETION_WAITING_DOTS="true"

plugins=(
  colored-man-pages
  last-working-dir
  zsh-autosuggestions
  zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

#######################################################################
# Fasd Configuration
#######################################################################
alias j="fasd_cd -d"
alias jj="fasd_cd -d -i"

fasd_cache="$HOME/.fasd-init-zsh"
if [ "$(command -v fasd)" -nt "$fasd_cache" -o ! -s "$fasd_cache" ]; then
  fasd --init auto >| "$fasd_cache"
fi
source "$fasd_cache"
unset fasd_cache

#######################################################################
# Miniconda Configuration
#######################################################################

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sasdata/ra/user/yuman.tam/.local/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sasdata/ra/user/yuman.tam/.local/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/sasdata/ra/user/yuman.tam/.local/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/sasdata/ra/user/yuman.tam/.local/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

#######################################################################
# Perl Configuration
#######################################################################
# Install module
# perl -MCPAN -e shell
export PERL5LIB="$HOME/.local/brew/lib/perl5/5.30.0"
export PERL="$HOME/.local/bin/perl"
