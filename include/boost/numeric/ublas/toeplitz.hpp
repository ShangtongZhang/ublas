//
//  Copyright (c) 2016
//  Shangtong Zhang
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef _BOOST_UBLAS_TOEPLITZ_
#define _BOOST_UBLAS_TOEPLITZ_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/detail/temporary.hpp>

namespace boost { namespace numeric { namespace ublas {
    
    /** \brief A toeplitz matrix of values of type \c T.
     *
     * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
     * \tparam A the type of Storage array. Default is \c unbounded_array
     */
    template<class T, class A = unbounded_array<T> >
    class toeplitz_matrix:
        public matrix_container<toeplitz_matrix<T, A> > {
            typedef T* pointer;
            typedef toeplitz_matrix<T, A> self_type;
        public:
            typedef typename A::size_type size_type;
            typedef typename A::difference_type difference_type;
            typedef T value_type;
            typedef const T& const_reference;
            typedef T& reference;
            typedef A array_type;
            typedef packed_tag storage_category;
            typedef row_major_tag orientation_category;
            typedef const matrix_reference<const self_type> const_closure_type;
            typedef matrix_reference<self_type> closure_type;
        private:
        public:
            // Construction and destruction
            BOOST_UBLAS_INLINE
            toeplitz_matrix ():
                matrix_container<self_type>(),
                size1_(0), size2_(0), data_(0) {}
            BOOST_UBLAS_INLINE
            toeplitz_matrix (size_type size1, size_type size2):
                matrix_container<self_type>(),
                size1_(size1), size2_(size2), data_(size1 + size2 - 1) {}
            BOOST_UBLAS_INLINE
            toeplitz_matrix (size_type size1, size_type size2, const array_type& data):
                matrix_container<self_type>(),
                size1_(size1), size2_(size2), data_(data) {}
            BOOST_UBLAS_INLINE
            toeplitz_matrix (const toeplitz_matrix& m):
                matrix_container<self_type>(),
                size1_(m.size1_), size2_(m.size_2),data_(m.data_) {}
           
            // Initialize toeplitz matrix with first row and first column
            template<class ARRAY>
            BOOST_UBLAS_INLINE
            toeplitz_matrix (const ARRAY& row, const ARRAY& column):
                matrix_container<self_type>(),
                size1_(column.size()), size2_(row.size()), data_(row.size() + column.size() - 1) {
                    BOOST_UBLAS_CHECK(row.size(), external_logic());
                    BOOST_UBLAS_CHECK(column.size(), external_logic());
                    // Ther first element of the row and the column must be the same.
                    BOOST_UBLAS_CHECK(row[0] == column[0], external_logic());
                    std::copy(column.rbegin(), column.rend(), data ().begin());
                    std::copy(row.begin() + 1, row.end(), data ().begin() + column.size());
            }
            
            // Accessors
            BOOST_UBLAS_INLINE
            size_type size1() const {
                return size1_;
            }
            BOOST_UBLAS_INLINE
            size_type size2() const {
                return size2_;
            }
            
            // Storage accessors
            BOOST_UBLAS_INLINE
            const array_type& data() const {
                return data_;
            }
            BOOST_UBLAS_INLINE
            array_type& data() {
                return data_;
            }
            
            BOOST_UBLAS_INLINE
            void resize (size_type size1, size_type size2, bool preserve = true) {
                if (preserve) {
                    BOOST_UBLAS_CHECK(size1 + size2 == size1_ + size2_, external_logic());
                    size1_ = size1;
                    size2_ = size2;
                } else {
                    size1_ = size1;
                    size2_ = size2;
                    data ().resize(size1 + size2 - 1);
                }
            }
            
            /** 
             * Element access
             * In toeplitz matrix, once an element is changed, all the elements 
             * in the corresponding diagonal line will be changed.
             */
            BOOST_UBLAS_INLINE
            const_reference operator () (size_type i, size_type j) const {
                BOOST_UBLAS_CHECK (i < size1_, bad_index ());
                BOOST_UBLAS_CHECK (j < size2_, bad_index ());
                if (i >= j) {
                    return data () [size1_ - 1 - i + j];
                } else {
                    return data () [size1_ + j - i - 1];
                }
            }
            
            BOOST_UBLAS_INLINE
            reference operator () (size_type i, size_type j) {
                BOOST_UBLAS_CHECK (i < size1_, bad_index ());
                BOOST_UBLAS_CHECK (j < size2_, bad_index ());
                if (i >= j) {
                    return data () [size1_ - 1 - i + j];
                } else {
                    return data () [size1_ + j - i - 1];
                }
            }
            
            // Element assignment
            BOOST_UBLAS_INLINE
            reference insert_element (size_type i, size_type j, const_reference t) {
                return (operator () (i, j) = t);
            }
            BOOST_UBLAS_INLINE
            void erase_element (size_type i, size_type j) {
                operator () (i, j) = value_type();
            }
            
            // Zeroing
            BOOST_UBLAS_INLINE
            void clear () {
                std::fill (data ().begin(), data ().end(), value_type());
            }
            
            // Assignment
            BOOST_UBLAS_INLINE
            toeplitz_matrix& operator = (const toeplitz_matrix& m) {
                size1_ = m.size1_;
                size2_ = m.size2_;
                data () = m.data();
                return *this;
            }
            
            // Swapping
            BOOST_UBLAS_INLINE
            void swap (toeplitz_matrix& m) {
                if (this != &m) {
                    std::swap (size1_, m.size1_);
                    std::swap (size2_, m.size2_);
                    data ().swap (m.data ());
                }
            }
            
            BOOST_UBLAS_INLINE
            friend void swap (toeplitz_matrix& m1, toeplitz_matrix& m2) {
                m1.swap (m2);
            }
            
            // Iterator types
            class iterator1;
            class const_iterator1;
            class iterator2;
            class const_iterator2;
            
            typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
            typedef reverse_iterator_base1<iterator1> reverse_iterator1;
            typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
            typedef reverse_iterator_base2<iterator2> reverse_iterator2;
            
            // Iterators simply are indices.
            class const_iterator1:
            public container_const_reference<toeplitz_matrix>,
            public random_access_iterator_base<packed_random_access_iterator_tag, const_iterator1, value_type> {
            public:
                typedef typename toeplitz_matrix::value_type value_type;
                typedef typename toeplitz_matrix::difference_type difference_type;
                typedef typename toeplitz_matrix::const_reference reference;
                typedef typename toeplitz_matrix::pointer pointer;
                
                typedef const_iterator2 dual_iterator_type;
                typedef const_reverse_iterator2 dual_reverse_iterator_type;
                
                // Construction and destruction
                BOOST_UBLAS_INLINE
                const_iterator1 ():
                    container_const_reference<self_type> (), it1_(), it2_() {};
                BOOST_UBLAS_INLINE
                const_iterator1 (const self_type&m, size_type it1, size_type it2):
                    container_const_reference<self_type> (m), it1_(it1), it2_(it2) {};
                BOOST_UBLAS_INLINE
                const_iterator1 (const iterator1& it):
                    container_const_reference<self_type> (it()), it1_(it.it1_), it2_(it.it2_) {}
                
                // Arithmetic
                BOOST_UBLAS_INLINE
                const_iterator1& operator ++ () {
                    ++ it1_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                const_iterator1& operator -- () {
                    -- it1_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                const_iterator1 &operator += (difference_type n) {
                    it1_ += n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                const_iterator1 &operator -= (difference_type n) {
                    it1_ -= n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                difference_type operator - (const const_iterator1 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it2_ == it.it2_, external_logic ());
                    return it1_ - it.it1_;
                }
                
                // Dereference
                BOOST_UBLAS_INLINE
                const_reference operator * () const {
                    return (*this) () (it1_, it2_);
                }
                BOOST_UBLAS_INLINE
                const_reference operator [] (difference_type n) const {
                    return *(*this + n);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator2 begin() const {
                    return const_iterator2((*this) (), it1_, 0);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator2 cbegin() const {
                    return begin();
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator2 end() const {
                    return const_iterator2((*this) (), it1_, (*this) ().size2());
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator2 cend() const {
                    return end();
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator2 rbegin () const {
                    return const_reverse_iterator2 (end ());
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator2 crbegin () const {
                    return rbegin ();
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator2 rend () const {
                    return const_reverse_iterator2 (begin ());
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator2 crend () const {
                    return rend ();
                }
                
                // Indices
                BOOST_UBLAS_INLINE
                size_type index1() const {
                    return it1_;
                }
                
                BOOST_UBLAS_INLINE
                size_type index2() const {
                    return it2_;
                }
                
                // Assignment
                BOOST_UBLAS_INLINE
                const_iterator1 &operator = (const const_iterator1 &it) {
                    container_const_reference<self_type>::assign (&it ());
                    it1_ = it.it1_;
                    it2_ = it.it2_;
                    return *this;
                }
                
                // Comparison
                BOOST_UBLAS_INLINE
                bool operator == (const const_iterator1 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it2_ == it.it2_, external_logic ());
                    return it1_ == it.it1_;
                }
                BOOST_UBLAS_INLINE
                bool operator < (const const_iterator1 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it2_ == it.it2_, external_logic ());
                    return it1_ < it.it1_;
                }
                
            private:
                size_type it1_;
                size_type it2_;
            };
            
            BOOST_UBLAS_INLINE
            const_iterator1 begin1 () const {
                return const_iterator1(*this, 0, 0);
            }
            BOOST_UBLAS_INLINE
            const_iterator1 cbegin1 () const {
                return begin1();
            }
            BOOST_UBLAS_INLINE
            const_iterator1 end1 () const {
                return const_iterator1(*this, size1_, 0);
            }
            BOOST_UBLAS_INLINE
            const_iterator1 cend1 () const {
                return end1 ();
            }
            
            
            class iterator1:
            public container_reference<toeplitz_matrix>,
            public random_access_iterator_base<packed_random_access_iterator_tag, iterator1, value_type> {
            public:
                typedef typename toeplitz_matrix::value_type value_type;
                typedef typename toeplitz_matrix::difference_type difference_type;
                typedef typename toeplitz_matrix::reference reference;
                typedef typename toeplitz_matrix::pointer pointer;
                
                typedef iterator2 dual_iterator_type;
                typedef reverse_iterator2 dual_reverse_iterator_type;
                
                // Construction and destruction
                BOOST_UBLAS_INLINE
                iterator1 ():
                container_reference<self_type> (), it1_(), it2_() {};
                BOOST_UBLAS_INLINE
                iterator1 (self_type&m, size_type it1, size_type it2):
                container_reference<self_type> (m), it1_(it1), it2_(it2) {};
                
                // Arithmetic
                BOOST_UBLAS_INLINE
                iterator1& operator ++ () {
                    ++ it1_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                iterator1& operator -- () {
                    -- it1_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                iterator1 &operator += (difference_type n) {
                    it1_ += n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                iterator1 &operator -= (difference_type n) {
                    it1_ -= n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                difference_type operator - (const iterator1 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it2_ == it.it2_, external_logic ());
                    return it1_ - it.it1_;
                }
                
                // Dereference
                BOOST_UBLAS_INLINE
                reference operator * () const {
                    return (*this) ()(it1_, it2_);
                }
                BOOST_UBLAS_INLINE
                reference operator [] (difference_type n) const {
                    return *(*this + n);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                iterator2 begin() const {
                    return iterator2((*this) (), it1_, 0);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                iterator2 end() const {
                    return iterator2((*this) (), it1_, (*this) ().size2());
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                reverse_iterator2 rbegin () const {
                    return reverse_iterator2 (end ());
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                reverse_iterator2 rend () const {
                    return reverse_iterator2 (begin ());
                }
                
                // Indices
                BOOST_UBLAS_INLINE
                size_type index1() const {
                    return it1_;
                }
                
                BOOST_UBLAS_INLINE
                size_type index2() const {
                    return it2_;
                }
                
                // Assignment
                BOOST_UBLAS_INLINE
                iterator1 &operator = (const iterator1 &it) {
                    container_reference<self_type>::assign (&it ());
                    it1_ = it.it1_;
                    it2_ = it.it2_;
                    return *this;
                }
                
                // Comparison
                BOOST_UBLAS_INLINE
                bool operator == (const iterator1 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it2_ == it.it2_, external_logic ());
                    return it1_ == it.it1_;
                }
                BOOST_UBLAS_INLINE
                bool operator < (const iterator1 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it2_ == it.it2_, external_logic ());
                    return it1_ < it.it1_;
                }
                
            private:
                size_type it1_;
                size_type it2_;
                
                friend class const_iterator1;
            };
            
            BOOST_UBLAS_INLINE
            iterator1 begin1 () {
                return iterator1(*this, 0, 0);
            }
            BOOST_UBLAS_INLINE
            iterator1 end1 () {
                return iterator1(*this, size1_, 0);
            }
            
            class const_iterator2:
            public container_const_reference<toeplitz_matrix>,
            public random_access_iterator_base<packed_random_access_iterator_tag, const_iterator2, value_type> {
            public:
                typedef typename toeplitz_matrix::value_type value_type;
                typedef typename toeplitz_matrix::difference_type difference_type;
                typedef typename toeplitz_matrix::const_reference reference;
                typedef typename toeplitz_matrix::pointer pointer;
                
                typedef const_iterator1 dual_iterator_type;
                typedef const_reverse_iterator1 dual_reverse_iterator_type;
                
                // Construction and destruction
                BOOST_UBLAS_INLINE
                const_iterator2 ():
                container_const_reference<self_type> (), it1_(), it2_() {};
                BOOST_UBLAS_INLINE
                const_iterator2 (const self_type&m, size_type it1, size_type it2):
                container_const_reference<self_type> (m), it1_(it1), it2_(it2) {};
                BOOST_UBLAS_INLINE
                const_iterator2 (const iterator2& it):
                container_const_reference<self_type> (it()), it1_(it.it1_), it2_(it.it2_) {}
                
                // Arithmetic
                BOOST_UBLAS_INLINE
                const_iterator2& operator ++ () {
                    ++ it2_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                const_iterator2& operator -- () {
                    -- it2_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                const_iterator2 &operator += (difference_type n) {
                    it2_ += n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                const_iterator2 &operator -= (difference_type n) {
                    it2_ -= n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                difference_type operator - (const const_iterator2 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it1_ == it.it1_, external_logic ());
                    return it2_ - it.it2_;
                }
                
                // Dereference
                BOOST_UBLAS_INLINE
                const_reference operator * () const {
                    return (*this) () (it1_, it2_);
                }
                BOOST_UBLAS_INLINE
                const_reference operator [] (difference_type n) const {
                    return *(*this + n);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator1 begin() const {
                    return const_iterator1((*this) (), 0, it2_);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator1 cbegin() const {
                    return begin();
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator1 end() const {
                    return const_iterator1((*this) (), (*this) ().size1(), it2_);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_iterator1 cend() const {
                    return end();
                }
               
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator1 rbegin () const {
                    return const_reverse_iterator1 (end ());
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator1 crbegin () const {
                    return rbegin ();
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator1 rend () const {
                    return const_reverse_iterator1 (begin ());
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                const_reverse_iterator1 crend () const {
                    return rend ();
                }
                
                // Indices
                BOOST_UBLAS_INLINE
                size_type index1() const {
                    return it1_;
                }
                
                BOOST_UBLAS_INLINE
                size_type index2() const {
                    return it2_;
                }
                
                // Assignment
                BOOST_UBLAS_INLINE
                const_iterator2 &operator = (const const_iterator2 &it) {
                    container_const_reference<self_type>::assign (&it ());
                    it1_ = it.it1_;
                    it2_ = it.it2_;
                    return *this;
                }
                
                // Comparison
                BOOST_UBLAS_INLINE
                bool operator == (const const_iterator2 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it1_ == it.it1_, external_logic ());
                    return it2_ == it.it2_;
                }
                BOOST_UBLAS_INLINE
                bool operator < (const const_iterator2 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it1_ == it.it1_, external_logic ());
                    return it2_ < it.it2_;
                }
                
            private:
                size_type it1_;
                size_type it2_;
            };
            
            BOOST_UBLAS_INLINE
            const_iterator2 begin2 () const {
                return const_iterator2(*this, 0, 0);
            }
            BOOST_UBLAS_INLINE
            const_iterator2 cbegin2 () const {
                return begin2();
            }
            BOOST_UBLAS_INLINE
            const_iterator2 end2 () const {
                return const_iterator2(*this, 0, size2_);
            }
            BOOST_UBLAS_INLINE
            const_iterator2 cend2 () const {
                return end2 ();
            }
            
            
            class iterator2:
            public container_reference<toeplitz_matrix>,
            public random_access_iterator_base<packed_random_access_iterator_tag, iterator2, value_type> {
            public:
                typedef typename toeplitz_matrix::value_type value_type;
                typedef typename toeplitz_matrix::difference_type difference_type;
                typedef typename toeplitz_matrix::reference reference;
                typedef typename toeplitz_matrix::pointer pointer;
                
                typedef iterator1 dual_iterator_type;
                typedef reverse_iterator1 dual_reverse_iterator_type;
                
                // Construction and destruction
                BOOST_UBLAS_INLINE
                iterator2 ():
                container_reference<self_type> (), it1_(), it2_() {};
                BOOST_UBLAS_INLINE
                iterator2 (self_type&m, size_type it1, size_type it2):
                container_reference<self_type> (m), it1_(it1), it2_(it2) {};
                
                // Arithmetic
                BOOST_UBLAS_INLINE
                iterator2& operator ++ () {
                    ++ it2_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                iterator2& operator -- () {
                    -- it2_;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                iterator2 &operator += (difference_type n) {
                    it2_ += n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                iterator2 &operator -= (difference_type n) {
                    it2_ -= n;
                    return *this;
                }
                BOOST_UBLAS_INLINE
                difference_type operator - (const iterator2 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it1_ == it.it1_, external_logic ());
                    return it2_ - it.it2_;
                }
                
                // Dereference
                BOOST_UBLAS_INLINE
                reference operator * () const {
                    return (*this) ()(it1_, it2_);
                }
                BOOST_UBLAS_INLINE
                reference operator [] (difference_type n) const {
                    return *(*this + n);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                iterator1 begin() const {
                    return iterator1((*this) (), 0, it2_);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                iterator1 end() const {
                    return iterator1((*this) (), (*this) ().size1(), it2_);
                }
                
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                reverse_iterator1 rbegin () const {
                    return reverse_iterator1 (end ());
                }
                BOOST_UBLAS_INLINE
#ifdef BOOST_UBLAS_MSVC_NESTED_CLASS_RELATION
                typename self_type::
#endif
                reverse_iterator1 rend () const {
                    return reverse_iterator1 (begin ());
                }
                
                // Indices
                BOOST_UBLAS_INLINE
                size_type index1() const {
                    return it1_;
                }
                
                BOOST_UBLAS_INLINE
                size_type index2() const {
                    return it2_;
                }
                
                // Assignment
                BOOST_UBLAS_INLINE
                iterator2 &operator = (const iterator2 &it) {
                    container_reference<self_type>::assign (&it ());
                    it1_ = it.it1_;
                    it2_ = it.it2_;
                    return *this;
                }
                
                // Comparison
                BOOST_UBLAS_INLINE
                bool operator == (const iterator2 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it1_ == it.it1_, external_logic ());
                    return it2_ == it.it2_;
                }
                BOOST_UBLAS_INLINE
                bool operator < (const iterator2 &it) const {
                    BOOST_UBLAS_CHECK (&(*this) () == &it (), external_logic ());
                    BOOST_UBLAS_CHECK (it1_ == it.it1_, external_logic ());
                    return it2_ < it.it2_;
                }
                
            private:
                size_type it1_;
                size_type it2_;
                
                friend class const_iterator2;
            };
            
            BOOST_UBLAS_INLINE
            iterator2 begin2 () {
                return iterator2(*this, 0, 0);
            }
            BOOST_UBLAS_INLINE
            iterator2 end2 () {
                return iterator2(*this, 0, size2_);
            }
            
            // Reverse iterators
            
            BOOST_UBLAS_INLINE
            const_reverse_iterator1 rbegin1 () const {
                return const_reverse_iterator1 (end1 ());
            }
            BOOST_UBLAS_INLINE
            const_reverse_iterator1 crbegin1 () const {
                return rbegin1 ();
            }
            BOOST_UBLAS_INLINE
            const_reverse_iterator1 rend1 () const {
                return const_reverse_iterator1 (begin1 ());
            }
            BOOST_UBLAS_INLINE
            const_reverse_iterator1 crend1 () const {
                return rend1 ();
            }
            
            BOOST_UBLAS_INLINE
            reverse_iterator1 rbegin1 () {
                return reverse_iterator1 (end1 ());
            }
            BOOST_UBLAS_INLINE
            reverse_iterator1 rend1 () {
                return reverse_iterator1 (begin1 ());
            }
            
            BOOST_UBLAS_INLINE
            const_reverse_iterator2 rbegin2 () const {
                return const_reverse_iterator2 (end2 ());
            }
            BOOST_UBLAS_INLINE
            const_reverse_iterator2 crbegin2 () const {
                return rbegin2 ();
            }
            BOOST_UBLAS_INLINE
            const_reverse_iterator2 rend2 () const {
                return const_reverse_iterator2 (begin2 ());
            }
            BOOST_UBLAS_INLINE
            const_reverse_iterator2 crend2 () const {
                return rend2 ();
            }
            
            BOOST_UBLAS_INLINE
            reverse_iterator2 rbegin2 () {
                return reverse_iterator2 (end2 ());
            }
            BOOST_UBLAS_INLINE
            reverse_iterator2 rend2 () {
                return reverse_iterator2 (begin2 ());
            }
            
        private:
            A data_;
            size_type size1_;
            size_type size2_;
        };
}}}

#endif