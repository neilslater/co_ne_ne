# RuNeNe

[![Gem Version](https://badge.fury.io/rb/ru_ne_ne.png)](http://badge.fury.io/rb/ru_ne_ne)
[![Build Status](https://travis-ci.org/neilslater/ru_ne_ne.png?branch=master)](http://travis-ci.org/neilslater/ru_ne_ne)
[![Coverage Status](https://coveralls.io/repos/neilslater/ru_ne_ne/badge.png?branch=master)](https://coveralls.io/r/neilslater/ru_ne_ne?branch=master)
[![Inline docs](http://inch-ci.org/github/neilslater/ru_ne_ne.png?branch=master)](http://inch-ci.org/github/neilslater/ru_ne_ne)
[![Code Climate](https://codeclimate.com/github/neilslater/ru_ne_ne.png)](https://codeclimate.com/github/neilslater/ru_ne_ne)
[![Dependency Status](https://gemnasium.com/neilslater/ru_ne_ne.png)](https://gemnasium.com/neilslater/ru_ne_ne)

*Ru*by *Ne*ural *Ne*tworks.

*Please note this gem is effectively abandoned before any first version was published.*

The proto-gem code is being left online for reference. It is MIT licensed, so anyone is free to fork
it or to just take code snippets and use in their own projects.

If you are looking for a neural-network library, you have a few better choices than attempting
to complete this unfinished gem. Here are some:

 * [Tensor Flow for Ruby.](https://github.com/somaticio/tensorflow.rb)

 * ruby-fann gem. The FANN library is somewhat behind on new developments in deep networks, but is
   perfectly servicable for small-to-medium two or three layer feed-forward networks.

 * Learn enough Python to use one of the many great options in that language, such as Keras. Python has
   far better machine learning libraries than Ruby, and will likely continue to do so for many years.

## Features

 * Uses NArray sfloat arrays to represent inputs, outputs and weights.

## Installation

Add this line to your application's Gemfile:

    gem 'ru_ne_ne'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install ru_ne_ne


## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
